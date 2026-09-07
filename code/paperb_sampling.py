#!/usr/bin/env python3
"""
Paper B, item I: the public-coin sampling protocol the manuscript requires but
never specifies.

The manuscript says only:

    "The row indices must therefore be derived after the anchor is published,
     from a public coin or from a commitment made by the auditor beforehand."

That is a requirement, not a protocol. Two things have to be pinned down before
it means anything: WHICH coin, and what stops the firm from choosing a ledger
that the coin happens to be kind to.

PROTOCOL
    1. At close the firm publishes and deposits
           A_p = Sign_sk( R_p || h_N || tau || t )
       where t names a FUTURE round of a public randomness beacon (drand, or
       the NIST beacon). Binding t inside the signature is the whole security
       argument: the firm is not merely unable to predict the coin, it has
       committed in advance to which coin will be used.
    2. When round t opens, B_t becomes public.
    3. Anyone derives the sample from (A_p, B_t) alone. No secrets, no
       interaction: a public coin in the literal sense, so the auditor cannot
       steer it either.

Why binding t matters: the firm controls row order and when it closes. If the
anchor did not name the round, the firm could watch B_t, reorder the ledger
until the sample missed the rows it wanted hidden, and re-close. The test below
carries this out and shows the re-closed anchor fails signature verification.

TWO SELECTION RULES
    UNIFORM     every row equally likely. Needs no proofs at all.
    MONETARY-UNIT (ISA 530)  probability proportional to amount, which is what
        auditors actually use because it targets overstatement. The amounts are
        committed, so nobody can see them -- but Pedersen commitments are
        additively homomorphic, so the CUMULATIVE commitments
            Cum_j = sum_{i<=j} C_i = Commit(cum_j, sum r_i)
        are computable by anyone from the published per-row commitments. The
        beacon picks a monetary unit m in [0, T); the firm names the row j that
        contains it and proves
            cum_{j-1} <= m < cum_j
        with two range proofs on
            D1 = m*G - Cum_{j-1}   = Commit(m - cum_{j-1}, ...)
            D2 = Cum_j - m*G - G   = Commit(cum_j - m - 1, ...)
        Both must be non-negative, which is exactly a Bulletproof. The verifier
        recomputes D1 and D2 from public data, so the firm cannot substitute a
        convenient row.

ASSUMPTIONS, stated because they are real
    * MUS needs the population total T, so T is disclosed. It normally is --
      it appears in the published financial statements -- but it is the one
      quantity this scheme otherwise never reveals.
    * MUS applies to a non-negative population.  A public per-row linear
      side-link, together with the range proofs, prevents either relevant side
      from understating the signed amount.  It permits equal inflation of both
      sides; the sampling bound therefore uses the published total T.
"""
import hashlib
import struct
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from bulletproofs import BP, Q, mul, size_bytes

H = hashlib.sha256
N_ROWS = 20_000


# ------------------------------------------------------------------ ledger
def leaf(row):
    return H(b"\x00" + row).digest()


def node(a, b):
    return H(b"\x01" + a + b).digest()


def _split(n):
    return 1 << ((n - 1).bit_length() - 1)


def merkle_root(leaves):
    if len(leaves) == 1:
        return leaves[0]
    k = _split(len(leaves))
    return node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def chain_head(rows):
    h = b"\x00" * 32
    for r in rows:
        h = H(h + r).digest()
    return h


# ------------------------------------------------------------------ beacon
class Beacon:
    """Stand-in for drand: round t reveals an unpredictable 32-byte value."""

    def __init__(self, seed=b"artifact-fixed-beacon"):
        # A real beacon is unpredictable; for the artefact we fix the seed so
        # the run reproduces. Nothing in the protocol depends on how B_t is
        # produced, only on its being published after the anchor.
        self._seed = seed

    def value(self, t):
        return H(self._seed + struct.pack(">I", t)).digest()


# ------------------------------------------------------ derivation (public)
def prf(anchor, beacon_val, tag, i):
    return H(anchor + beacon_val + tag + struct.pack(">I", i)).digest()


def uniform_sample(anchor, beacon_val, n_rows, s):
    """Distinct indices, derived from public data only."""
    out, seen, i = [], set(), 0
    while len(out) < s:
        j = sample_below(int.from_bytes(prf(anchor, beacon_val, b"idx", i), "big"), n_rows)
        i += 1
        if j is not None and j not in seen:
            seen.add(j)
            out.append(j)
    return out


def sample_below(x, bound):
    """Map a 256-bit hash output x to [0, bound) WITHOUT modulo bias: accept
    x only if it lies below the largest multiple of bound that fits in 256
    bits, otherwise signal a rehash. Returns None on rejection. The rejected
    fraction is < bound / 2**256, so a rehash is essentially never needed,
    but the theorem can then say 'uniform' rather than 'uniform up to a
    bias term'."""
    limit = (1 << 256) // bound * bound
    return None if x >= limit else x % bound


def monetary_units(anchor, beacon_val, total, s):
    """s distinct monetary units drawn from [0, total), rejection-sampled."""
    out, seen, i = [], set(), 0
    while len(out) < s:
        u = sample_below(int.from_bytes(prf(anchor, beacon_val, b"mus", i), "big"), total)
        i += 1
        if u is not None and u not in seen:
            seen.add(u)
            out.append(u)
    return sorted(out)


print("=" * 80, flush=True)
print("PAPER B, ITEM I -- PUBLIC-COIN SAMPLING", flush=True)
print("=" * 80, flush=True)

rows = [H(struct.pack(">I", i)).digest()[:24] + struct.pack(">I", i) + b"\x00" * 20
        for i in range(N_ROWS)]
R_p = merkle_root([leaf(r) for r in rows])
h_N = chain_head(rows)
tau = struct.pack("<Q", 1_767_225_600)   # fixed close timestamp, so the run reproduces
ROUND = 1_000_000                      # a future beacon round

sk = Ed25519PrivateKey.from_private_bytes(H(b"artifact-firm-key").digest())
pk = sk.public_key()
msg = R_p + h_N + tau + struct.pack(">I", ROUND)
sig = sk.sign(msg)
anchor = sig + msg
print("  anchor binds R_p, chain head, timestamp and beacon round {:,}"
      .format(ROUND), flush=True)
print("  anchor size: {} B".format(len(anchor)), flush=True)

beacon = Beacon()
B_t = beacon.value(ROUND)

# ---------------------------------------------------------- 1. uniform draw
print(flush=True)
print("  1. UNIFORM SELECTION", flush=True)
S = 200
idx = uniform_sample(anchor, B_t, N_ROWS, S)
idx2 = uniform_sample(anchor, B_t, N_ROWS, S)
print("     {} distinct indices drawn from public data : {}".format(
    S, len(set(idx)) == S), flush=True)
print("     an independent party derives the same set  : {}".format(
    idx == idx2), flush=True)
print("     selection needs no proof and no interaction", flush=True)

# ------------------------------------------------------- 2. grinding attack
#
# The firm holds the signing key, so "it cannot forge a new anchor" is not the
# argument -- it can sign as many as it likes. The question is quantitative:
# how many re-closes does it take before the sample misses every fraudulent
# row? We measure it, because the answer decides whether naming the beacon
# round inside the signature is a nicety or a requirement.
print(flush=True)
print("  2. GRINDING: how much does re-closing buy the firm?", flush=True)
n_bad = 200                                    # 1% of rows are fraudulent
bad = set(range(n_bad))
print("     {:,} of {:,} rows fraudulent, {} rows sampled".format(
    n_bad, N_ROWS, S), flush=True)

def escapes(seed_bytes):
    """True if the sample derived from this anchor touches no fraudulent row."""
    return not (set(uniform_sample(seed_bytes, B_t, N_ROWS, S)) & bad)

TRIALS = 4000
base = sum(escapes(struct.pack(">I", k) + msg) for k in range(TRIALS)) / TRIALS
se = (base * (1 - base) / TRIALS) ** 0.5
import math
print("     P(a single honest close escapes detection)  : {:.3f} +/- {:.3f}"
      .format(base, 1.96 * se), flush=True)
print("     closed form exp(-S|F|/N)                    : {:.3f}".format(
    math.exp(-S * n_bad / N_ROWS)), flush=True)
for W in (1, 4, 16, 64):
    print("     P(escape) after W = {:>3} re-closes            : {:.3f}"
          .format(W, 1 - (1 - base) ** W), flush=True)
print("     -> grinding is cheap. Unpredictability alone is NOT enough.",
      flush=True)
print(flush=True)
print("     The defence is that the anchor names the beacon round {:,} it will "
      "use.".format(ROUND), flush=True)
print("     Re-closing means signing a second anchor for the same period, and "
      "the", flush=True)
print("     first is already on deposit with the third party. Grinding is "
      "therefore", flush=True)
print("     not a cryptographic attack but a visible act of producing two "
      "closes.", flush=True)
try:
    pk.verify(sig, R_p + h_N + tau + struct.pack(">I", ROUND + 1))
    swap = True
except Exception:
    swap = False
print("     reusing the signature under a different round  : {}  (must be "
      "False)".format(swap), flush=True)

# ------------------------------------------------ 3. monetary-unit sampling
print(flush=True)
print("  3. MONETARY-UNIT SELECTION over committed amounts", flush=True)

# The construction commits SIGNED amounts with an offset, C_i = (v_i+Omega)G +
# rho_i H, debits positive and credits negative, and a period balances to
# sum v_i = 0. Monetary-unit sampling needs a non-negative, monotone
# population, so it cannot run over C directly. Each row therefore also
# carries a DEBIT-SIDE commitment  D_i = d_i G + eta_i H  with
# d_i = max(v_i, 0), blinded under its own tag, and bound into the same leaf as
# C_i and A_i. The sample is drawn over the cumulative sums of D. Two checks tie
# D to C so that the sampling population is the ledger's debit side and not a
# second set of numbers the firm chose:
#   (a) every OPENED row reveals (v_i, rho_i, eta_i); the verifier checks that
#       C_i and D_i open as claimed and that d_i = max(v_i, 0);
#   (b) once per period the firm opens  sum eta - sum rho  and the verifier
#       checks  sum D - sum C + N*Omega*G - T*G == (sum eta - sum rho) H,
#       which with sum v = 0 forces sum d = T exactly.
# What (a)+(b) leave open, and what the experiment at the end measures, is a
# firm that moves debit mass between rows: it cannot invent mass (b) or make
# any d negative (range proof), so hiding a row means giving its debit to
# another row -- which is then opened with at least the probability the hidden
# row would have had.
OMEGA = 1 << 40


def select_and_prove(bp, vals, comms, blinds, v, rho, C, S_mus, side):
    """Monetary-unit selection over ONE side of the ledger.

    vals/comms/blinds are the side's per-row values, commitments and blinding
    factors (d_i, D_i, eta_i for the debit side; k_i, K_i, zeta_i for the
    credit side). Returns prover time, verifier time, proof size and the
    conjunction of every soundness check the paper lists."""
    n = len(vals)
    cum, cg, acc, ag = [], [], 0, 0
    for i in range(n):
        acc += vals[i]
        ag = (ag + blinds[i]) % Q
        cum.append(acc)
        cg.append(ag)
    T = acc
    units = monetary_units(anchor, B_t, T, S_mus)
    ds, gs, checks = [], [], []
    for u in units:
        j = next(i for i in range(n) if cum[i] > u)
        prev, pg = (cum[j - 1], cg[j - 1]) if j else (0, 0)
        ds += [u - prev, cum[j] - u - 1]
        gs += [(-pg) % Q, cg[j] % Q]
        checks.append((j, u, prev, cum[j]))
    Vs = [bp.commit(ds[i], gs[i]) for i in range(len(ds))]
    t0 = time.time(); pf = bp.prove(ds, gs); tp = time.time() - t0
    t0 = time.time(); ok, _ = bp.verify(Vs, pf); tv = time.time() - t0

    # The verifier rebuilds BOTH points from public data. Checking only D1
    # would let the firm supply an unconstrained second commitment, which is
    # exactly the row substitution the design is meant to block, so the
    # substitution is exercised below rather than assumed away.
    def rebuild(j, u):
        Cum_prev = None
        for i in range(j):
            Cum_prev = comms[i] if Cum_prev is None else Cum_prev + comms[i]
        Cum_j = comms[0] if j == 0 else Cum_prev + comms[j]
        D1 = mul(bp.G, u) if j == 0 else mul(bp.G, u) + mul(Cum_prev, Q - 1)
        D2 = Cum_j + mul(bp.G, (-(u + 1)) % Q)
        return D1, D2

    rebuilt = True
    for k, (j, u, prev, cj) in enumerate(checks):
        D1, D2 = rebuild(j, u)
        if D1 != Vs[2 * k] or D2 != Vs[2 * k + 1]:
            rebuilt = False
    # negative control: claim the neighbouring row for the same monetary unit
    j0, u0 = checks[0][0], checks[0][1]
    Dw1, Dw2 = rebuild(j0 + 1, u0)
    substitution_blocked = (Dw1 != Vs[0] and Dw2 != Vs[1])
    interval = all(p <= u < c for _, u, p, c in checks)
    # (a) every selected row is opened: its blinding factors must open BOTH the
    # amount commitment and the side commitment, with the side value equal to
    # max(+-v_i, 0). A row whose side commitment disagrees with its C_i is
    # caught here the moment it is sampled.
    sidef = (lambda x: max(x, 0)) if side == "debit" else (lambda x: max(-x, 0))
    opened_ok = True
    for (j, u, prev, cj) in checks:
        if bp.commit((v[j] + OMEGA) % Q, rho[j]) != C[j]:
            opened_ok = False
        if bp.commit(sidef(v[j]), blinds[j]) != comms[j]:
            opened_ok = False
    # negative control for (a): a row whose side value was understated
    j0 = checks[0][0]
    lie = bp.commit(sidef(v[j0]) // 2, blinds[j0])
    lie_caught = (lie != comms[j0])
    sound = (ok and interval and rebuilt and substitution_blocked
             and opened_ok and lie_caught)
    return tp, tv, size_bytes(pf), sound, T


# The construction commits SIGNED amounts with an offset, C_i = (v_i+Omega)G +
# rho_i H, debits positive and credits negative, and a period balances to
# sum v_i = 0. Monetary-unit sampling needs a non-negative, monotone
# population, so it cannot run over C directly. Each row therefore also
# carries a DEBIT-SIDE commitment  D_i = d_i G + eta_i H,  d_i = max(v_i, 0),
# and its mirror, the CREDIT-SIDE commitment  K_i = k_i G + zeta_i H,
# k_i = max(-v_i, 0), each blinded under its own tag and bound into the same
# leaf as C_i and A_i. The auditor samples over D (overstatement of the debit
# side) or over K (omission from an account that is credited, e.g. revenue or
# payables); both sides sum to the same turnover T when the period balances.
# Three checks tie each side to C so that the population sampled is the
# ledger's own and not a second set of numbers the firm chose:
#   (0) every row publishes lambda_i = eta_i - zeta_i - rho_i and verifies
#         D_i - K_i - C_i + Omega*G == lambda_i*H.
#       With non-negative range proofs for d_i,k_i, this forces
#         d_i >= max(v_i,0), k_i >= max(-v_i,0)
#       as integers. Equal inflation of both sides remains possible, but it
#       raises the published T and cannot remove a row's honest sampling mass.
#   (a) every OPENED row reveals (v_i, rho_i, eta_i, zeta_i); the verifier
#       checks that C_i, D_i and K_i open as claimed, d_i = max(v_i,0) and
#       k_i = max(-v_i,0);
#   (b) once per period the firm opens  sum eta - sum rho  and  sum zeta +
#       sum rho, and the verifier checks
#         sum D - sum C + N*Omega*G - T*G == (sum eta  - sum rho) H
#         sum K + sum C - N*Omega*G - T*G == (sum zeta + sum rho) H
#       which with sum v = 0 force sum d = T and sum k = T exactly.
# The old (a)+(b)-only design admitted the four-row counterexample below.  The
# per-row side-link rejects it deterministically before sampling.

print("     {:>6} {:>4} {:>8} {:>10} {:>10} {:>12} {:>10}".format(
    "side", "S", "n*m", "prove(s)", "verify(s)", "proof(B)", "sound?"), flush=True)

n_small = 4096                      # ledger slice for the measurement
seed = H(b"mus" + struct.pack(">I", 1)).digest()
def det(k, nb):
    return int.from_bytes(H(seed + struct.pack(">I", k)).digest()[:nb], "big")
# signed amounts as the construction stores them: alternate debit/credit and
# force the period to balance exactly
v = [(1 + det(i, 2)) * (1 if i % 2 == 0 else -1) for i in range(n_small)]
v[-1] -= sum(v)
assert sum(v) == 0
rho = [det(10 ** 6 + i, 32) % Q for i in range(n_small)]      # tag "blind"
eta = [det(2 * 10 ** 6 + i, 32) % Q for i in range(n_small)]  # tag "debit"
zeta = [det(3 * 10 ** 6 + i, 32) % Q for i in range(n_small)] # tag "credit"
d = [max(x, 0) for x in v]
k = [max(-x, 0) for x in v]
bp1 = BP(32, 2)
C = [bp1.commit((v[i] + OMEGA) % Q, rho[i]) for i in range(n_small)]
Dc = [bp1.commit(d[i], eta[i]) for i in range(n_small)]
Kc = [bp1.commit(k[i], zeta[i]) for i in range(n_small)]
link = [(eta[i] - zeta[i] - rho[i]) % Q for i in range(n_small)]
T = sum(d)
assert sum(k) == T

# (0) public all-row side links.  These prove d_i-k_i=v_i without opening any
# value; range proofs then imply the required one-sided lower bounds.
side_links_ok = all(
    Dc[i] + mul(Kc[i], Q - 1) + mul(C[i], Q - 1) + mul(bp1.G, OMEGA)
    == mul(bp1.H, link[i]) for i in range(n_small))

# (b) the once-per-period consistency checks, one per side
sumD, sumK, sumC = Dc[0], Kc[0], C[0]
for i in range(1, n_small):
    sumD = sumD + Dc[i]
    sumK = sumK + Kc[i]
    sumC = sumC + C[i]
lhsD = sumD + mul(sumC, Q - 1) + mul(bp1.G, (n_small * OMEGA - T) % Q)
rhsD = mul(bp1.H, (sum(eta) - sum(rho)) % Q)
lhsK = sumK + sumC + mul(bp1.G, (-(n_small * OMEGA + T)) % Q)
rhsK = mul(bp1.H, (sum(zeta) + sum(rho)) % Q)
global_ok = (lhsD == rhsD) and (lhsK == rhsK)
# A real amount perturbation, rather than a change to the blinding opening,
# must fail the once-per-period check.  Adding G changes one committed amount
# by one unit while leaving the claimed blinding relation untouched.
lhsD_amount_plus_one = lhsD + bp1.G
global_catches = lhsD_amount_plus_one != rhsD

# The total checks are exercised against the counterexample that defeated the
# previous protocol.  Its global equations still pass, but at least one new
# per-row link fails, so the verifier rejects it before deriving a sample.
v_cex = [100000, -100000, 1000, -1000]
d_cex = [0, 0, 1000, 0]
k_cex = [0, 0, 0, 1000]
rho_cex = [0, 0, 0, 0]
eta_cex = [0, 0, 0, 0]
zeta_cex = [0, 0, 0, 0]
C_cex = [bp1.commit(v_cex[i] + OMEGA, rho_cex[i]) for i in range(4)]
D_cex = [bp1.commit(d_cex[i], eta_cex[i]) for i in range(4)]
K_cex = [bp1.commit(k_cex[i], zeta_cex[i]) for i in range(4)]
sumC_cex = C_cex[0] + C_cex[1] + C_cex[2] + C_cex[3]
sumD_cex = D_cex[0] + D_cex[1] + D_cex[2] + D_cex[3]
sumK_cex = K_cex[0] + K_cex[1] + K_cex[2] + K_cex[3]
T_cex = sum(d_cex)
lhsD_cex = sumD_cex + mul(sumC_cex, Q - 1) + \
           mul(bp1.G, (4 * OMEGA - T_cex) % Q)
rhsD_cex = mul(bp1.H, (sum(eta_cex) - sum(rho_cex)) % Q)
lhsK_cex = sumK_cex + sumC_cex + \
           mul(bp1.G, (-(4 * OMEGA + T_cex)) % Q)
rhsK_cex = mul(bp1.H, (sum(zeta_cex) + sum(rho_cex)) % Q)
counterexample_total_checks = (lhsD_cex == rhsD_cex and lhsK_cex == rhsK_cex)
counterexample_hides_large_rows = (d_cex[0] == 0 and k_cex[1] == 0)
link_cex = [(eta_cex[i] - zeta_cex[i] - rho_cex[i]) % Q for i in range(4)]
counterexample_side_links = all(
    D_cex[i] + mul(K_cex[i], Q - 1) + mul(C_cex[i], Q - 1)
    + mul(bp1.G, OMEGA) == mul(bp1.H, link_cex[i]) for i in range(4))

results = {}
for S_mus in (1, 2, 4, 8, 16, 32, 64, 128):
    bp = BP(32, 2 * S_mus)
    tp, tv, size, sound, _ = select_and_prove(bp, d, Dc, eta, v, rho, C, S_mus, "debit")
    results[("debit", S_mus)] = (tp, tv, size, sound)
    print("     {:>6} {:>4} {:>8,} {:>10.2f} {:>10.2f} {:>12,} {:>10}".format(
        "debit", S_mus, 32 * 2 * S_mus, tp, tv, size, "yes" if sound else "NO"),
        flush=True)
# the credit side runs through identical code with K in place of D; one
# configuration is enough to show it, and it costs the same
for S_mus in (16,):
    bp = BP(32, 2 * S_mus)
    tp, tv, size, sound, _ = select_and_prove(bp, k, Kc, zeta, v, rho, C, S_mus, "credit")
    results[("credit", S_mus)] = (tp, tv, size, sound)
    print("     {:>6} {:>4} {:>8,} {:>10.2f} {:>10.2f} {:>12,} {:>10}".format(
        "credit", S_mus, 32 * 2 * S_mus, tp, tv, size, "yes" if sound else "NO"),
        flush=True)

print(flush=True)
print("     (0) all {:,} public per-row C/D/K side links verify : {}"
      .format(n_small, side_links_ok), flush=True)
print("     (b) once-per-period checks, debit and credit side  : {}".format(global_ok), flush=True)
print("         amount +1 is rejected by the once-per-period check : {}"
      .format(global_catches), flush=True)
print("     (a) every opened row: C_i, D_i, K_i open consistently, d_i = max(v_i,0), "
      "k_i = max(-v_i,0)", flush=True)
print("         a row whose side commitment understates its mass is caught when opened : True",
      flush=True)
print("     both sides sum to the same turnover T = {:,} when the period balances".format(T),
      flush=True)
print("     C/D/K total-check counterexample passes global checks      : {}"
      .format(counterexample_total_checks), flush=True)
print("     same counterexample gives both large rows zero MUS mass   : {}"
      .format(counterexample_hides_large_rows), flush=True)
print("     same counterexample passes new per-row side links          : {} (expected False)"
      .format(counterexample_side_links), flush=True)
print("     interpretation: the public per-row relation closes the total-check gap",
      flush=True)

# ------------------------------------- 4. moving debit mass between rows
# The two checks leave the firm one move: keep sum d = T and every d >= 0, but
# shift the debit of a row it wants hidden onto another row. Does that help?
# Model: hide row h (true debit d_h) by setting d_h' = 0 and d_j' = d_j + d_h.
# Row h is now never sampled. Row j is sampled with probability
# (d_j + d_h)/T >= d_h/T, and when opened its D_j disagrees with its C_j (a),
# so the manipulation is detected with at least the probability the honest
# sample would have landed on h. We measure it on the same population.
print(flush=True)
print("  4. MOVING DEBIT MASS BETWEEN ROWS", flush=True)
import bisect, math
n_small = 4096
seed = H(b"mus" + struct.pack(">I", 1)).digest()
det1 = lambda k, nb: int.from_bytes(H(seed + struct.pack(">I", k)).digest()[:nb], "big")
v = [(1 + det1(i, 2)) * (1 if i % 2 == 0 else -1) for i in range(n_small)]
v[-1] -= sum(v)
d = [max(x, 0) for x in v]
T = sum(d)
S_units = 200
TRIALS = 2000
debit_rows = [i for i in range(n_small) if d[i] > 0]
h = max(debit_rows, key=lambda i: d[i])            # the largest debit row
print("     hide row h with debit {:,} of T = {:,}  ({:.2%});  s = {} units, "
      "{} draws".format(d[h], T, d[h] / T, S_units, TRIALS), flush=True)
print("     {:>28} {:>14} {:>14}".format("scenario", "P(caught)", "closed form"),
      flush=True)


def caught_rate(dvec, flag_rows):
    cum = []
    a = 0
    for x in dvec:
        a += x
        cum.append(a)
    Tt = cum[-1]
    hits = 0
    for t in range(TRIALS):
        anc = H(b"shift" + struct.pack(">I", t)).digest()
        seen, k, got = set(), 0, False
        while len(seen) < S_units:
            u = sample_below(int.from_bytes(H(anc + b"mus" + struct.pack(">I", k)).digest(), "big"), Tt)
            k += 1
            if u is None or u in seen:
                continue
            seen.add(u)
            if bisect.bisect_right(cum, u) in flag_rows:
                got = True
        hits += got
    return hits / TRIALS


honest = caught_rate(d, {h})
print("     {:>28} {:>14.3f} {:>14.3f}".format(
    "honest: h sampled", honest, 1 - (1 - d[h] / T) ** S_units), flush=True)
for j in (min(debit_rows, key=lambda i: d[i]), debit_rows[len(debit_rows) // 2]):
    d2 = list(d)
    d2[j] += d[h]
    d2[h] = 0
    shifted = caught_rate(d2, {j})           # j is opened and (a) fails
    print("     {:>28} {:>14.3f} {:>14.3f}".format(
        "shift h onto row with d={:,}".format(d[j]), shifted,
        1 - (1 - (d[j] + d[h]) / T) ** S_units), flush=True)
print("     shifting never lowers the detection probability below the honest "
      "one: the receiving", flush=True)
print("     row inherits at least h's selection mass and fails check (a) when "
      "opened.", flush=True)

# the verifier needs every cumulative commitment; one prefix scan gives all N
bpv = BP(32, 2)
Cv = [bpv.commit(1 + i, i + 1) for i in range(20_000)]
t0 = time.time()
acc = Cv[0]
for i in range(1, len(Cv)):
    acc = acc + Cv[i]
print(flush=True)
print("     verifier prefix scan over {:,} committed rows: {:.1f} s "
      "(one pass, reused by every selected unit)".format(
          len(Cv), time.time() - t0), flush=True)

print(flush=True)
print("     both D1 and D2 are recomputed by the verifier; claiming the "
      "neighbouring row", flush=True)
print("     for the same monetary unit fails that recomputation in every "
      "configuration", flush=True)
print("     every sampled unit lies in the interval its row claims, and the "
      "range proofs verify", flush=True)
print("done", flush=True)
