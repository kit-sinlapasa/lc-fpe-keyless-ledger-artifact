#!/usr/bin/env python3
"""
Paper B primitives: tamper-evident ledger -- implementation and measurement.

Implemented for real:
  * hash-chained journal   h_i = SHA256(h_{i-1} || row_i)
  * Merkle tree per period, inclusion proofs
  * period-close anchor    Ed25519 signature over (root || h_last || timestamp)
  * selective disclosure   open s sampled rows with Merkle paths
  * tamper detection tests

Range proofs are NOT exercised by this benchmark.  They are implemented and
validated separately in bulletproofs.py, which measures aggregated proofs up to
n*m = 1024; the size figures quoted for a full partition follow the published
formula 32*(9 + 2k) with k = log2(n*m) -- one proof for the whole partition,
not one per row.
"""
import os
import time
import hashlib
import struct
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

N = 20000
H = hashlib.sha256


def leaf(row):
    return H(b"\x00" + row).digest()


def node(a, b):
    return H(b"\x01" + a + b).digest()


def _split(n):
    """largest power of two strictly less than n (the k of RFC 6962)"""
    return 1 << ((n - 1).bit_length() - 1)


def build_merkle(leaves):
    """RFC 6962 (Certificate Transparency) Merkle tree.

    An earlier version padded an odd level by repeating its last node.  That
    makes the root malleable: a ledger of n rows and one of n+1 rows whose last
    row is duplicated hash to the SAME root, so an adversary could double-post
    the final row of a period and the signed anchor would still verify.  It is
    the flaw recorded as CVE-2012-2459 in Bitcoin.  RFC 6962 splits at the
    largest power of two below n and never duplicates, so the tree shape is
    fixed by n; with the leaf/internal domain separation above the construction
    is second-preimage resistant.

    Returns (tree, root) where tree is (hash, left, right).
    """
    def rec(lv):
        if len(lv) == 1:
            return (lv[0], None, None)
        k = _split(len(lv))
        L, R = rec(lv[:k]), rec(lv[k:])
        return (node(L[0], R[0]), L, R)
    tr = rec(leaves)
    return tr, tr[0]


def merkle_path(tree, idx, n):
    """RFC 6962 audit path, leaf side first."""
    path = []

    def rec(t, i, m):
        if m == 1:
            return
        k = _split(m)
        if i < k:
            rec(t[1], i, k)
            path.append((t[2][0], 0))
        else:
            rec(t[2], i - k, m - k)
            path.append((t[1][0], 1))

    rec(tree, idx, n)
    return path


def verify_path(lf, path, root):
    cur = lf
    for sib, right in path:
        cur = node(sib, cur) if right else node(cur, sib)
    return cur == root


print("=" * 78, flush=True)
print("PAPER B PRIMITIVES -- tamper-evident ledger, {:,} rows".format(N),
      flush=True)
print("=" * 78, flush=True)

# A leaf must bind everything the auditor will later be shown or will verify
# against: the RowID, the encrypted fields, all four commitments, and the
# public per-row side-link scalar. Anchoring
# a row that omits C_i and A_i would let the firm keep one set of commitments
# for the auditor and another for itself -- the anchored ledger and the
# committed ledger would be two objects, and only one of them would be fixed.
# Here the commitments are 33-byte stand-ins of the right size; the real ones
# are produced by the Pedersen layer benchmarked separately.
def make_row(i):
    rid = struct.pack("<IIQI", 1, 202609, 10 ** 6 + i, i % 7 + 1)   # e,p,doc,line
    enc = struct.pack("<Iq", i * 7 % 9973, (-1) ** i * (i + 1)).ljust(24, b"\x00")
    c_i = H(b"C" + rid).digest() + b"\x02"                          # 33 B  signed amount
    a_i = H(b"A" + rid).digest() + b"\x03"                          # 33 B  account code
    d_i = H(b"D" + rid).digest() + b"\x02"                          # 33 B  debit side, for sampling
    k_i = H(b"K" + rid).digest() + b"\x03"                          # 33 B  credit side, mirror of D
    link_i = H(b"L" + rid).digest()                                 # 32 B  C/D/K side link
    return rid + enc + c_i + a_i + d_i + k_i + link_i


rows = [make_row(i) for i in range(N)]
ROW_BYTES = len(rows[0])

# --- build chain + Merkle
t0 = time.time()
h = b"\x00" * 32
for r in rows:
    h = H(h + r).digest()
t_chain = time.time() - t0

t0 = time.time()
leaves = [leaf(r) for r in rows]
tree, root = build_merkle(leaves)
t_merkle = time.time() - t0

depth = (N - 1).bit_length()
print("  hash chain build      : {:7.3f} s   ({:.1f} us/row)"
      .format(t_chain, t_chain / N * 1e6), flush=True)
print("  Merkle build          : {:7.3f} s   ({:.1f} us/row), depth {}"
      .format(t_merkle, t_merkle / N * 1e6, depth), flush=True)

# --- anchor: a canonical signed envelope, then the custodian's countersignature.
# Every field a verifier needs to reject a replay, a cross-period substitution
# or an ambiguous parse is inside the signed bytes, not beside them.
sk_F = Ed25519PrivateKey.generate()          # the firm's key (adversary holds it)
sk_C = Ed25519PrivateKey.generate()          # the custodian's key (it does not)
pk_F, pk_C = sk_F.public_key(), sk_C.public_key()
prev_anchor = H(b"anchor of period p-1").digest()
env = (b"LEDGER-ANCHOR/1"                    # schema/algorithm version (15 B)
       + struct.pack("<II", 1, 202609)       # entity, period
       + struct.pack("<I", N)                # row count
       + root + h                            # Merkle root, chain head
       + prev_anchor                         # hash of the previous anchor
       + struct.pack("<Q", int(time.time()))  # firm's close timestamp tau
       + struct.pack("<I", 1_000_000))       # beacon round t named in advance
t0 = time.time()
sig_F = sk_F.sign(env)
t_sign = time.time() - t0
anchor = env + sig_F
# custodian receipt: countersigns the anchor together with its own receipt time
receipt = anchor + struct.pack("<Q", int(time.time()) + 60)
sig_C = sk_C.sign(receipt)
anchor_bytes = len(anchor)
anchor_full = len(receipt) + len(sig_C)
print("  period-close anchor   : {:7.3f} ms  firm-signed {} B; with custodian "
      "receipt {} B  (per PERIOD, not row)"
      .format(t_sign * 1e3, anchor_bytes, anchor_full), flush=True)
# the signature must cover every field: flip the beacon round and check
bad_env = env[:-4] + struct.pack("<I", 1_000_001)
try:
    pk_F.verify(sig_F, bad_env); reround = True
except Exception:
    reround = False
print("  anchor with a different beacon round verifies: {}  (expected False)"
      .format(reround), flush=True)
sig, pk, msg = sig_F, pk_F, env   # names used by the tamper tests below

# --- inclusion proofs / selective disclosure
S = 200
def _below(bound):
    """rejection-sampled uniform index: no modulo bias"""
    limit = (1 << 32) // bound * bound
    while True:
        x = int.from_bytes(os.urandom(4), "big")
        if x < limit:
            return x % bound


idxs = [_below(N) for _ in range(S)]
t0 = time.time()
proofs = [merkle_path(tree, i, N) for i in idxs]
t_prove = (time.time() - t0) / S
t0 = time.time()
ok = all(verify_path(leaves[idxs[j]], proofs[j], root) for j in range(S))
t_verify = (time.time() - t0) / S
proof_bytes = depth * 32
print("  inclusion proof       : {:7.1f} us gen, {:.1f} us verify, {} B each"
      .format(t_prove * 1e6, t_verify * 1e6, proof_bytes), flush=True)
print("  all {} proofs verify   : {}".format(S, ok), flush=True)
print("  selective disclosure of {} rows: {:.1f} KB total  ({} B row + {} B path each)"
      .format(S, S * (ROW_BYTES + proof_bytes) / 1024, ROW_BYTES, proof_bytes), flush=True)

# --- tamper detection
print("", flush=True)
print("  TAMPER TESTS", flush=True)
bad = list(rows)
bad[137] = bad[137][:16] + b"\xff" + bad[137][17:]

h2 = b"\x00" * 32
for r in bad:
    h2 = H(h2 + r).digest()
print("   modify 1 row  -> chain head differs : {}".format(h2 != h), flush=True)

leaves2 = [leaf(r) for r in bad]
_, root2 = build_merkle(leaves2)
print("   modify 1 row  -> Merkle root differs: {}".format(root2 != root),
      flush=True)

# The tampered ledger's envelope: same fields, but the root and chain head it
# would need. The firm's signature over the ORIGINAL envelope must not cover
# it. Only InvalidSignature counts as "rejected"; any other exception is a bug
# in this test and must surface. (An earlier version of this test referenced
# a variable that no longer existed, and its bare `except Exception` turned
# the resulting NameError into a passing result.)
from cryptography.exceptions import InvalidSignature
env_tampered = env.replace(root, root2).replace(h, h2)
assert env_tampered != env
try:
    pk.verify(sig, env_tampered)
    sig_ok = True
except InvalidSignature:
    sig_ok = False
print("   anchor signature still verifies     : {}  (expected False)"
      .format(sig_ok), flush=True)
assert not sig_ok

drop = rows[:5000] + rows[5001:]
_, root3 = build_merkle([leaf(r) for r in drop])
print("   delete 1 row  -> Merkle root differs: {}".format(root3 != root),
      flush=True)

dup = rows + [rows[-1]]
_, root_dup = build_merkle([leaf(r) for r in dup])
print("   duplicate last row -> root differs  : {}".format(root_dup != root),
      flush=True)

ins = rows[:5000] + [b"\xab" * ROW_BYTES] + rows[5000:]
_, root5 = build_merkle([leaf(r) for r in ins])
h5 = b"\x00" * 32
for r in ins:
    h5 = H(h5 + r).digest()
print("   insert 1 row  -> Merkle root differs: {}".format(root5 != root),
      flush=True)
print("   insert 1 row  -> chain head differs : {}".format(h5 != h),
      flush=True)

swap = list(rows)
swap[10], swap[11] = swap[11], swap[10]
_, root4 = build_merkle([leaf(r) for r in swap])
print("   reorder 2 rows-> Merkle root differs: {}".format(root4 != root),
      flush=True)

# --- storage
print("", flush=True)
print("  STORAGE OVERHEAD", flush=True)
print("   chain hashes are recomputable and need not be stored", flush=True)
print("   anchor: {} B per period  ->  {:.4f} B/row at N={:,}"
      .format(anchor_bytes, anchor_bytes / N, N), flush=True)

# --- range proofs: cited sizes vs the sampling alternative
print("", flush=True)
print("  RANGE PROOFS: aggregated Bulletproofs vs probabilistic sampling",
      flush=True)
import math
for m_ in (1, 1000, 20000):
    k = math.ceil(math.log2(64 * m_))
    size = 32 * (9 + 2 * k)
    print("   aggregate over {:>6,} amounts: {:>6,} B total  ({:.3f} B/row)"
          .format(m_, size, size / m_), flush=True)
print("   sampling instead: a wraparound forgery needs only ~2 corrupted rows;",
      flush=True)
for S_ in (100, 200, 1000):
    p = 1 - (1 - 2 / N) ** S_
    print("     sampling {:>5} of {:,} rows detects it with p = {:.1%}"
          .format(S_, N, p), flush=True)

# --------------------------------------------------------- keyless balance
# The manuscript reports the time to verify sum C_i == N*Omega*G + R*H with
# no key. That figure was quoted for several drafts without any script
# producing it -- the placeholder commitments above are hash digests, not
# curve points, so this file could not have measured it. Here it is measured
# on real P-256 commitments over the same N rows, reusing the generators and
# the hash-to-curve routine of bulletproofs.py so that every script in the
# artefact commits under the same G and H.
from bulletproofs import h2c, mul, Q as _ORDER
from Crypto.PublicKey import ECC

_G = h2c(b"BP/G")
_Hp = h2c(b"BP/H")
_OMEGA = 1 << 40

print("", flush=True)
print("  KEYLESS BALANCE over real commitments (N = {:,})".format(N), flush=True)
_amt = [(-1) ** i * (i % 9973 + 1) for i in range(N)]
_amt[-1] -= sum(_amt)                       # the period balances exactly
assert sum(_amt) == 0
_rho = [int.from_bytes(H(b"rho" + struct.pack(">I", i)).digest(), "big") % _ORDER
        for i in range(N)]
_R = sum(_rho) % _ORDER

t0 = time.time()
_C = [mul(_G, (_amt[i] + _OMEGA) % _ORDER) + mul(_Hp, _rho[i]) for i in range(N)]
t_commit = time.time() - t0

t0 = time.time()
_acc = _C[0]
for _pt in _C[1:]:
    _acc += _pt
_rhs = mul(_G, (N * _OMEGA) % _ORDER) + mul(_Hp, _R)
_ok = (_acc == _rhs)
t_balance = time.time() - t0

# and it must fail when a single satang is off
_bad = _C[0] + _G
_accb = _bad
for _pt in _C[1:]:
    _accb += _pt
_catches = (_accb != _rhs)

print("   build {:,} commitments : {:7.3f} s  (writer, once per period)"
      .format(N, t_commit), flush=True)
print("   verify sum C == N*Omega*G + R*H : {:7.3f} s   holds: {}"
      .format(t_balance, _ok), flush=True)
print("   the same check against a ledger off by one unit : {}  (must be True)"
      .format(_catches), flush=True)
print("     arithmetic only; sound verification adds the range-proof", flush=True)
print("     verification, which we do not measure at this size", flush=True)
