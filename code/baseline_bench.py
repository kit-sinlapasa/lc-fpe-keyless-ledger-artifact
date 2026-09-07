#!/usr/bin/env python3
"""
BASELINE COMPARISON for Paper A, Table tab:base -- rebuilt end to end.

Five schemes on the same 20,000-row synthetic ledger:
  0  plaintext                     reference
  1  AES-256-XTS, whole database   disk-level; one tweak, decrypt checked
  2  AES-256-GCM, per record       per-row AEAD; decrypt checked
  3  per-field FF1 (4 digits)      the naive FPE baseline; round trip checked
  4  LC-FPE (ours)                 the construction of Section 4, every row:
                                   RowID + registry, rejection-sampled alias,
                                   lifted FF1 under tweak (e,p,pi), Pedersen
                                   commitment with an independent PRF-derived
                                   blinding, AES-GCM for the amount and the
                                   text with derived nonces and AAD, then the
                                   keyless close and a full decrypt round trip

What changed from the version this replaces, and why it had to:
  * XTS encrypted and decrypted under two different random tweaks and then
    computed the trial balance from the ORIGINAL plaintext. The decrypt was
    never checked; it could not have been correct.
  * LC-FPE timed 1,200 rows and multiplied by N; reused a pool of 256 blinding
    points (a repeated blinding leaks amount differences to anyone); never ran
    the AEAD at all; and placed aliases in contiguous ranges rather than
    through the RowID-keyed selection the paper specifies.
  * Three different LC-FPE figures were in circulation (8.8, 10.3, 15.0 s)
    because the table was typed by hand. This script writes the LaTeX rows
    itself, to results/tab_base_rows.tex, and the paper takes them verbatim.

Timing discipline: one warm-up pass, then repeated passes; median and p95
are reported with the repetition count. Every scheme's decrypt is checked
for equality with the plaintext before its time is accepted.
"""
import os
import sys
import time
import struct
import hashlib
import platform
import statistics
import numpy as np
from collections import defaultdict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from Crypto.PublicKey import ECC
from Crypto.Math.Numbers import Integer

import lcfpe_impl as L

N, M = 20000, 200
ROW = 32
REPS_FAST, REPS_SLOW = 20, 5
ENTITY, PERIOD, PART, LEDGER = 1, 202609, 1, "GL"
OMEGA = 1 << 40

# ------------------------------------------------------------------ ledger
rng = np.random.default_rng(11)
w = 1.0 / np.arange(1, M + 1)
f = w / w.sum()
acct = rng.choice(M, size=N, p=f)
cc = rng.integers(0, 100, N)
dt = rng.integers(0, 10, N)
amt = rng.integers(-10 ** 7, 10 ** 7, N)
amt[-1] = -int(amt[:-1].sum())               # the period balances exactly
text = [("memo%05d" % i).encode() for i in range(N)]   # 9-byte free text

rows = [struct.pack("<HHHqqh", int(acct[i]), int(cc[i]), int(dt[i]),
                    int(amt[i]), i, 0).ljust(ROW, b"\x00") for i in range(N)]
blob = b"".join(rows)


def stats(samples):
    s = sorted(samples)
    return statistics.median(s), s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def timed(fn, reps):
    fn()                                      # warm-up, discarded
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return stats(out) + (reps,)


print("=" * 92, flush=True)
print("BASELINE COMPARISON   ledger = {:,} rows, {} accounts, {} B/row plaintext"
      .format(N, M, ROW), flush=True)
print("  {} | Python {} | {}".format(platform.platform(), platform.python_version(),
                                     platform.processor() or "cpu n/a"), flush=True)
print("  warm-up + {} reps (fast schemes) / {} reps (LC-FPE); median and p95"
      .format(REPS_FAST, REPS_SLOW), flush=True)
print("=" * 92, flush=True)

res = {}

# ------------------------------------------------------------ 0 plaintext
def pt_tb():
    bal = defaultdict(int)
    for i in range(N):
        bal[int(acct[i])] += int(amt[i])
    return bal


def pt_close():
    return sum(int(x) for x in amt) == 0


tb_med, tb_p95, r1 = timed(pt_tb, REPS_FAST)
cl_med, cl_p95, _ = timed(pt_close, REPS_FAST)
res["Plaintext"] = dict(enc=(0.0, 0.0), tb=(tb_med, tb_p95), close=(cl_med, cl_p95),
                        bytes=ROW, keyless="---", reps=r1, ok=True)

# ------------------------------------------------------------- 1 AES-XTS
k_xts = os.urandom(64)
tweak_xts = os.urandom(16)                    # ONE tweak, used for both directions
ct_xts = None


def xts_enc():
    global ct_xts
    e = Cipher(algorithms.AES(k_xts), modes.XTS(tweak_xts)).encryptor()
    ct_xts = e.update(blob) + e.finalize()


def xts_dec_tb():
    d = Cipher(algorithms.AES(k_xts), modes.XTS(tweak_xts)).decryptor()
    pt = d.update(ct_xts) + d.finalize()
    assert pt == blob, "XTS decrypt does not round-trip"
    bal = defaultdict(int)
    for i in range(N):
        a, _, _, v, _, _ = struct.unpack_from("<HHHqqh", pt, i * ROW)
        bal[a] += v
    return bal


e_med, e_p95, r = timed(xts_enc, REPS_FAST)
tb_med, tb_p95, _ = timed(xts_dec_tb, REPS_FAST)
res["AES-256-XTS (whole DB)"] = dict(enc=(e_med, e_p95), tb=(tb_med, tb_p95),
                                     close=(tb_med, tb_p95), bytes=ROW,
                                     keyless="no", reps=r, ok=True)

# ------------------------------------------------------------- 2 AES-GCM
k_gcm = os.urandom(32)
gcm = AESGCM(k_gcm)
ct_gcm = None


def gcm_enc():
    global ct_gcm
    out = []
    for i in range(N):
        nonce = os.urandom(12)
        out.append((nonce, gcm.encrypt(nonce, rows[i], None)))
    ct_gcm = out


def gcm_dec_tb():
    bal = defaultdict(int)
    for i in range(N):
        nonce, c = ct_gcm[i]
        pt = gcm.decrypt(nonce, c, None)
        assert pt == rows[i]
        a, _, _, v, _, _ = struct.unpack_from("<HHHqqh", pt, 0)
        bal[a] += v
    return bal


e_med, e_p95, r = timed(gcm_enc, REPS_FAST)
tb_med, tb_p95, _ = timed(gcm_dec_tb, REPS_FAST)
res["AES-256-GCM (per record)"] = dict(enc=(e_med, e_p95), tb=(tb_med, tb_p95),
                                       close=(tb_med, tb_p95),
                                       bytes=ROW + 12 + 16, keyless="no",
                                       reps=r, ok=True)

# -------------------------------------------------------- 3 per-field FF1
ff1_field = L.FF1(os.urandom(32), radix=10)
ct_field = None


def ff1_field_enc():
    global ct_field
    ct_field = [ff1_field.encrypt([int(ch) for ch in "{:04d}".format(int(acct[i]))],
                                  tweak=b"acct") for i in range(N)]


def ff1_field_check():
    for i in range(0, N, 97):                  # spot-check the round trip
        pt = ff1_field.decrypt(ct_field[i], tweak=b"acct")
        assert int("".join(map(str, pt))) == int(acct[i])


e_med, e_p95, r = timed(ff1_field_enc, REPS_SLOW)
ff1_field_check()
res["Per-field FF1 (4 digits)"] = dict(enc=(e_med, e_p95), tb=None, close=None,
                                       bytes=ROW, keyless="no", reps=r, ok=True)

# -------------------------------------------------------------- 4 LC-FPE
K = L.ALPHA - M
k = L.alloc_hamilton(f, K)                    # exact l1 minimiser (alloc_opt)
start = np.concatenate([[0], np.cumsum(k)[:-1]])
owner = np.repeat(np.arange(M), k)            # alias -> account, for decrypt
key_alias, key_blind, key_nonce = os.urandom(32), os.urandom(32), os.urandom(32)
key_ff1, key_aead = os.urandom(32), os.urandom(32)
ff1 = L.FF1(key_ff1, radix=10)
aead = AESGCM(key_aead)
TWEAK = struct.pack("<IIH", ENTITY, PERIOD, PART)     # (e, p, pi)

curve = ECC._curves["p256"]
G = ECC.EccPoint(int(curve.Gx), int(curve.Gy), curve="p256")
ORDER, P_, B_ = int(curve.order), int(curve.p), int(curve.b)


def h2c(label):
    ctr = 0
    while True:
        h = hashlib.sha256(label + ctr.to_bytes(4, "big")).digest()
        x = int.from_bytes(h, "big") % P_
        rhs = (pow(x, 3, P_) - 3 * x + B_) % P_
        y = pow(rhs, (P_ + 1) // 4, P_)
        if pow(y, 2, P_) == rhs:
            return ECC.EccPoint(x, y, curve="p256")
        ctr += 1


Hp = h2c(b"LC-FPE/pedersen/H/v1")
enc_rows = None
rho_sum = 0


def lcfpe_enc():
    """The whole pipeline for every row, exactly as Algorithm 1 specifies."""
    global enc_rows, rho_sum
    reg = L.RowIDRegistry()
    out = []
    rsum = 0
    for i in range(N):
        rid = reg.check(L.row_id(ENTITY, PERIOD, PART, LEDGER, 10 ** 6 + i, 1))
        a = int(acct[i])
        j = L.alias_index(key_alias, None, None, int(k[a]), rid=rid)
        alias = int(start[a]) + j
        digits = [int(ch) for ch in L.lift(alias, int(cc[i]), int(dt[i]))]
        fmt = ff1.encrypt(digits, tweak=TWEAK)                      # 9 digits
        fmt_b = bytes(fmt)
        rho = L.blinding_scalar(key_blind, rid, ORDER)
        rsum = (rsum + rho) % ORDER
        C = G * Integer((int(amt[i]) + OMEGA) % ORDER) + Hp * Integer(rho)
        nonce = L.aead_nonce(0, 0, 10 ** 6 + i, 1)      # amount field
        ct_amt = aead.encrypt(nonce, struct.pack("<q", int(amt[i])), rid + fmt_b)
        nonce2 = L.aead_nonce(1, 0, 10 ** 6 + i, 1)     # text field
        ct_txt = aead.encrypt(nonce2, text[i], rid)
        out.append((rid, fmt, C, ct_amt, ct_txt))
    enc_rows, rho_sum = out, rsum


def lcfpe_close_keyless():
    """sum C_i == (N*Omega) G + R H, with R opened by the key holder."""
    acc = enc_rows[0][2].copy()
    for _, _, C, _, _ in enc_rows[1:]:
        acc += C
    rhs = G * Integer((N * OMEGA) % ORDER) + Hp * Integer(rho_sum)
    assert acc == rhs, "keyless balance check failed"
    return True


def lcfpe_verify_row(i, row):
    """Dec on one row, including the commitment recomputation of Algorithm 1.

    The commitment is NOT authenticated by either AEAD tag: the amount's
    associated data binds the RowID and the format-preserved triple, and the
    text's binds the RowID, so an adversary who replaces C alone forges
    nothing. What closes that gap is that C is a deterministic function of
    data Dec already holds: rho is derived from the RowID and v comes out of
    the authenticated ciphertext, so Dec recomputes (v+Omega)G + rho*H and
    compares. Without this line a modified C is accepted, which is the
    counterexample to the integrity proposition; with it, any accepted row
    that Enc did not produce implies an AEAD forgery."""
    rid, fmt, C, ct_amt, ct_txt = row
    d = ff1.decrypt(fmt, tweak=TWEAK)
    s = "".join(map(str, d))
    alias, c_, d_ = int(s[:4]), int(s[4:7]), int(s[7:9])
    if not (owner[alias] == int(acct[i]) and c_ == int(cc[i]) and d_ == int(dt[i])):
        return False
    nonce = L.aead_nonce(0, 0, 10 ** 6 + i, 1)
    v = struct.unpack("<q", aead.decrypt(nonce, ct_amt, rid + bytes(fmt)))[0]
    if v != int(amt[i]):
        return False
    nonce2 = L.aead_nonce(1, 0, 10 ** 6 + i, 1)
    if aead.decrypt(nonce2, ct_txt, rid) != text[i]:
        return False
    rho = L.blinding_scalar(key_blind, rid, ORDER)
    if C != G * Integer((v + OMEGA) % ORDER) + Hp * Integer(rho):
        return False
    return True


def lcfpe_dec_check():
    """Full round trip for every row: FF1 decrypt, un-lift, alias -> account,
    AEAD decrypt with the same AAD, equality with the plaintext. The
    commitment recomputation costs two scalar multiplications per row, the
    same as encryption, so it is spot-checked on every 97th row here and
    exercised directly by the negative test below."""
    for i, (rid, fmt, C, ct_amt, ct_txt) in enumerate(enc_rows):
        d = ff1.decrypt(fmt, tweak=TWEAK)
        s = "".join(map(str, d))
        alias, c_, d_ = int(s[:4]), int(s[4:7]), int(s[7:9])
        assert owner[alias] == int(acct[i]) and c_ == int(cc[i]) and d_ == int(dt[i])
        nonce = L.aead_nonce(0, 0, 10 ** 6 + i, 1)
        v = struct.unpack("<q", aead.decrypt(nonce, ct_amt, rid + bytes(fmt)))[0]
        assert v == int(amt[i])
        nonce2 = L.aead_nonce(1, 0, 10 ** 6 + i, 1)
        assert aead.decrypt(nonce2, ct_txt, rid) == text[i]
        if i % 97 == 0:
            rho = L.blinding_scalar(key_blind, rid, ORDER)
            assert C == G * Integer((v + OMEGA) % ORDER) + Hp * Integer(rho), \
                "commitment does not match the authenticated amount"
    return True


e_med, e_p95, r = timed(lcfpe_enc, REPS_SLOW)
c_med, c_p95, _ = timed(lcfpe_close_keyless, REPS_SLOW)
t0 = time.perf_counter(); lcfpe_dec_check(); t_dec = time.perf_counter() - t0
rid0, fmt0, C0, ca0, ct0 = enc_rows[0]
lc_bytes = len(fmt0) + 33 + len(ca0) + len(ct0)     # nonces derived, not stored
res["LC-FPE (ours)"] = dict(enc=(e_med, e_p95), tb=None, close=(c_med, c_p95),
                            bytes=lc_bytes, keyless="yes", reps=r, ok=True)

# tamper: moving an amount ciphertext to another row must fail the AEAD
rid1, fmt1, _, _, _ = enc_rows[1]
try:
    aead.decrypt(L.aead_nonce(0, 0, 10 ** 6 + 1, 1), ca0, rid1 + bytes(fmt1))
    moved_ok = True
except Exception:
    moved_ok = False

# tamper: replacing the COMMITMENT alone forges no tag, so it is caught only
# by the recomputation in Dec. An earlier version of this script passed C into
# the round trip and never looked at it, and the integrity proposition it was
# meant to support was false as stated.
assert lcfpe_verify_row(0, enc_rows[0]), "honest row must verify"
_bad_C = (enc_rows[0][2] + G)                       # any other group element
commit_caught = not lcfpe_verify_row(
    0, (enc_rows[0][0], enc_rows[0][1], _bad_C, enc_rows[0][3], enc_rows[0][4]))

# ----------------------------------------------------------------- report
print("", flush=True)
print("  {:<26} {:>6} {:>18} {:>18} {:>18} {:>8}".format(
    "scheme", "B/row", "encrypt med/p95 s", "trial bal med/p95", "close med/p95",
    "keyless"), flush=True)
print("  " + "-" * 100, flush=True)


def fmt2(t):
    return "n/a" if t is None else "{:.3f}/{:.3f}".format(*t)


for name, rr in res.items():
    print("  {:<26} {:>6} {:>18} {:>18} {:>18} {:>8}".format(
        name, rr["bytes"], fmt2(rr["enc"]), fmt2(rr["tb"]), fmt2(rr["close"]),
        rr["keyless"]), flush=True)

print("", flush=True)
print("  CHECKS", flush=True)
print("   XTS decrypt == plaintext, one tweak both ways   : True", flush=True)
print("   GCM decrypt == plaintext for all rows           : True", flush=True)
print("   per-field FF1 round trip (spot-checked)         : True", flush=True)
print("   LC-FPE full decrypt round trip, {:,} rows        : True  ({:.1f} s)".format(N, t_dec), flush=True)
print("   LC-FPE keyless close  sum C == N*Omega*G + R*H  : True", flush=True)
print("   amount ciphertext moved to another row          : {}  (expected False)".format(moved_ok), flush=True)
print("   commitment replaced, ciphertexts untouched      : {}  (expected True)"
      .format(commit_caught), flush=True)
print("     (no AEAD tag is forged in that attack; Dec catches it by", flush=True)
print("      recomputing C from the RowID-derived rho and the amount)", flush=True)
print("   RowID registry rejects a duplicate              : ", end="", flush=True)
try:
    rg = L.RowIDRegistry(); x = L.row_id(1, 2, 0, "GL", 3, 4); rg.check(x); rg.check(x); print("False")
except L.DuplicateRowError:
    print("True", flush=True)
print("   LC-FPE bytes/row = {} fmt + 33 C + {} amt AEAD + {} text AEAD (nonces derived)".format(
    len(fmt0), len(ca0), len(ct0)), flush=True)

lc, gc = res["LC-FPE (ours)"], res["AES-256-GCM (per record)"]
print("", flush=True)
print("  LC-FPE encrypt {:.1f} s median = {:.0f} rows/s; {:.0f}x AES-GCM; {:.1f}x plaintext bytes"
      .format(lc["enc"][0], N / lc["enc"][0], lc["enc"][0] / gc["enc"][0], lc["bytes"] / ROW), flush=True)

# ---------------------------------------------- LaTeX rows, single source
# The runner executes this file from artifact/code/, where the results
# directory is ../results; run directly from sim/ it is ../artifact/results.
_here = os.path.dirname(os.path.abspath(__file__))
_cands = [os.path.join(_here, "..", "results"),
          os.path.join(_here, "..", "artifact", "results")]
_res = next((p for p in _cands if os.path.isdir(p)), _cands[0])
os.makedirs(_res, exist_ok=True)
tex_path = os.path.join(_res, "tab_base_rows.tex")
order = ["Plaintext", "AES-256-XTS (whole DB)", "AES-256-GCM (per record)",
         "Per-field FF1 (4 digits)", "LC-FPE (ours)"]
with open(tex_path, "w", encoding="utf-8") as fh:
    for name in order:
        rr = res[name]
        e = "{:.3f}\\,s".format(rr["enc"][0]) if rr["enc"][0] else "0.000\\,s"
        c = "---" if rr["close"] is None else "{:.3f}\\,s".format(rr["close"][0])
        ky = {"yes": "\\textbf{yes}", "no": "no", "---": "---"}[rr["keyless"]]
        label = "\\textbf{LC-FPE}" if name.startswith("LC-FPE") else name
        if name.startswith("LC-FPE"):
            e, c = "\\textbf{" + e + "}", "\\textbf{" + c + "}"
        fh.write("{:<26} & {} & {} & {} & {}\\\\\n".format(label, rr["bytes"], e, c, ky))
print("  LaTeX rows written to {}".format(os.path.relpath(tex_path)), flush=True)
print("done", flush=True)
