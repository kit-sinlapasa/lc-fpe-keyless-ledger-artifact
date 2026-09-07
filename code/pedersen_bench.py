#!/usr/bin/env python3
"""
Pedersen commitments for the amount layer -- implementation and benchmark
against the Paillier numbers already measured in lcfpe_impl.py.

C_i = m_i * G + r_i * H      (m in satang, debits +, credits -)

Auditor checks, with NO key at all:
    sum_i C_i  ==  R * H        where R = sum_i r_i is opened by the client
which holds exactly when sum_i m_i = 0, i.e. debits = credits.

H is derived by try-and-increment hashing so that log_G(H) is unknown.
Curve: NIST P-256 (pycryptodome's EccPoint arithmetic is native code).
"""
import os
import time
import hashlib
from Crypto.PublicKey import ECC
from Crypto.Math.Numbers import Integer

CURVE = "p256"
_c = ECC._curves[CURVE]
P = int(_c.p)
ORDER = int(_c.order)
B = int(_c.b)
G = ECC.EccPoint(int(_c.Gx), int(_c.Gy), curve=CURVE)


def hash_to_point(label):
    """try-and-increment: find a valid curve point from a label."""
    ctr = 0
    while True:
        h = hashlib.sha256(label + ctr.to_bytes(4, "big")).digest()
        x = int.from_bytes(h, "big") % P
        rhs = (pow(x, 3, P) - 3 * x + B) % P
        y = pow(rhs, (P + 1) // 4, P)          # p = 3 mod 4
        if pow(y, 2, P) == rhs:
            try:
                return ECC.EccPoint(x, y, curve=CURVE)
            except ValueError:
                pass
        ctr += 1


H = hash_to_point(b"LC-FPE/pedersen/H/v1")
print("=" * 78, flush=True)
print("PEDERSEN COMMITMENT -- AMOUNT LAYER BENCHMARK", flush=True)
print("=" * 78, flush=True)
print("  curve = NIST P-256   order bits = {}".format(ORDER.bit_length()), flush=True)
print("  H derived by hash-to-curve (log_G(H) unknown)", flush=True)


def commit(m, r):
    return (G * Integer(m % ORDER)) + (H * Integer(r % ORDER))


# ---------------------------------------------------------------- correctness
n_test = 500
rs, ms = [], []
half = n_test // 2
for i in range(half):
    v = int.from_bytes(os.urandom(5), "big") % 10 ** 7
    ms.append(v)
    ms.append(-v)                      # balanced: every debit has its credit
for _ in range(len(ms)):
    rs.append(int.from_bytes(os.urandom(32), "big") % ORDER)

cs = [commit(ms[i], rs[i]) for i in range(len(ms))]
acc = cs[0].copy()
for c in cs[1:]:
    acc += c
R = sum(rs) % ORDER
expect = H * Integer(R)
ok = (int(acc.x) == int(expect.x) and int(acc.y) == int(expect.y))
print("  balanced-ledger check  sum(C) == R*H : {}".format(ok), flush=True)

# tamper check: alter one amount by 1 satang
cs_bad = list(cs)
cs_bad[3] = commit(ms[3] + 1, rs[3])
acc_bad = cs_bad[0].copy()
for c in cs_bad[1:]:
    acc_bad += c
bad_detected = not (int(acc_bad.x) == int(expect.x))
print("  1-satang tamper detected             : {}".format(bad_detected), flush=True)

# ---------------------------------------------------------------- speed
S = 3000
rr = [int.from_bytes(os.urandom(32), "big") % ORDER for _ in range(S)]
mm = [int.from_bytes(os.urandom(5), "big") % 10 ** 7 for _ in range(S)]

t0 = time.time()
for i in range(S):
    commit(mm[i], rr[i])
t_naive = (time.time() - t0) / S
print("", flush=True)
print("  commit, naive (2 scalar mults)  : {:8.1f} us/row  {:>10,.0f} rows/s"
      .format(t_naive * 1e6, 1 / t_naive), flush=True)

# optimisation: pool of precomputed r*H (offline, exactly like the Paillier r^n pool)
POOL = 4096
t0 = time.time()
pool = [(r, H * Integer(r)) for r in rr[:POOL] if True][:POOL]
t_pool = time.time() - t0
print("  offline pool of {} r*H         : {:.1f}s (one-off)"
      .format(len(pool), t_pool), flush=True)

t0 = time.time()
for i in range(S):
    _, rH = pool[i % len(pool)]
    (G * Integer(mm[i] % ORDER)) + rH
t_pooled = (time.time() - t0) / S
print("  commit, pooled r*H              : {:8.1f} us/row  {:>10,.0f} rows/s"
      .format(t_pooled * 1e6, 1 / t_pooled), flush=True)

# homomorphic addition
t0 = time.time()
a = cs[0].copy()
for i in range(1, len(cs)):
    a += cs[i]
t_add = (time.time() - t0) / (len(cs) - 1)
print("  homomorphic add (point add)     : {:8.1f} us      {:>10,.0f} adds/s"
      .format(t_add * 1e6, 1 / t_add), flush=True)

# full Sigma verification for a 20k-row partition
NROWS = 20000
t0 = time.time()
acc2 = cs[0].copy()
for i in range(1, len(cs)):
    acc2 += cs[i]
per_add = (time.time() - t0) / (len(cs) - 1)
t0 = time.time()
H * Integer(R)
t_smul = time.time() - t0
t_verify = per_add * NROWS + t_smul
print("  Sigma-verify a {:,}-row partition : {:.2f} s   (no key needed)"
      .format(NROWS, t_verify), flush=True)

# ---------------------------------------------------------------- comparison
pc_bytes = 33            # compressed P-256 point
aes_bytes = 12 + 8 + 16  # nonce + int64 amount + GCM tag
codes = 9
plain = 4 + 3 + 2 + 8

print("", flush=True)
print("  " + "-" * 74, flush=True)
print("  COMPARISON WITH THE MEASURED PAILLIER NUMBERS", flush=True)
print("  " + "-" * 74, flush=True)
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "", "Paillier", "Pedersen", "ratio"), flush=True)
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "commit/encrypt (us/row)", "39.4", "{:.1f}".format(t_pooled * 1e6),
    "{:.1f}x".format(t_pooled * 1e6 / 39.4)), flush=True)
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "homomorphic add (us)", "44.0", "{:.1f}".format(t_add * 1e6),
    "{:.2f}x".format(t_add * 1e6 / 44.0)), flush=True)
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "bytes per amount", "512", str(pc_bytes + aes_bytes),
    "{:.2f}x".format((pc_bytes + aes_bytes) / 512)), flush=True)
tot_pai, tot_ped = codes + 512, codes + pc_bytes + aes_bytes
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "row expansion vs plaintext",
    "{:.1f}x".format(tot_pai / plain),
    "{:.1f}x".format(tot_ped / plain),
    "-"), flush=True)
print("  {:<26} {:>12} {:>12} {:>12}".format(
    "Sigma check needs a key?", "YES", "NO", "-"), flush=True)
print("", flush=True)
print("  1M-row ledger: plaintext {:.0f} MB | Paillier {:.0f} MB | Pedersen {:.0f} MB"
      .format(plain * 1e6 / 1e6, tot_pai * 1e6 / 1e6, tot_ped * 1e6 / 1e6), flush=True)
