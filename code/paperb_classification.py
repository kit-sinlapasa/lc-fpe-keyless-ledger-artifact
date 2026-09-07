#!/usr/bin/env python3
"""
Paper B, item 2: closing the per-account balance gap.

The construction verifies that a claimed set of rows sums to a claimed balance.
It never checks that the claimed set IS the account. A firm that misassigns
rows presents a balance that is arithmetically perfect and factually wrong, and
until now the paper declared that an open problem.

It splits into two directions, which need different tools and have very
different outcomes.

    SUBSTITUTION -- the claimed set contains a row that is not account a
        (swapping two rows between accounts is the clean case: both balances
        stay internally consistent and the total is untouched).
        Closed COMPLETELY and deterministically. Commit the account code per
        row, A_i = a_i*G + sigma_i*H. To claim row i for account a the firm
        reveals sigma_i; anyone checks A_i - a*G == sigma_i*H. A row that is
        not account a cannot pass, because that would open a Pedersen
        commitment two ways.

    OMISSION -- a row that IS account a is left out of the claimed set,
        understating the balance.
        Cannot be closed deterministically without revealing the account of
        every row, which would defeat the point. Bounded instead, and the
        bound is the useful part: with monetary-unit sampling the detection
        probability tracks the MAGNITUDE omitted, not the row count. Omitting
        rows worth M out of a population total T is caught with probability
        1 - (1 - M/T)^s. Material omissions are caught; immaterial ones are
        not; and that is the trade auditors already make.

The omission check is FREE. It rides on rows the vouching sample of
Section "Public-Coin Sampling" already opens: for each sampled row the auditor
sees the account, and a row claiming account a that is absent from the claimed
set is the evidence.

POPULATION, stated because the bound depends on it: the s monetary units are
drawn from the WHOLE ledger, not from the claimed set. Sampling inside the
claimed set says nothing about omissions by construction.

BLINDING: sigma_i is derived under a key separate from the amount blinding
K_blind, so opening an account never weakens an amount. Pedersen is perfectly
hiding given independent blinding; deriving sigma by PRF makes it
computationally hiding under PRF security. This matters more here than for
amounts because the account domain is small -- a few thousand codes -- and a
commitment over a small domain is brute-forceable the moment its blinding is
reused or predictable. That is the same small-domain problem the companion
paper exists to solve.
"""
import bisect
import hashlib
import struct
import time
from bulletproofs import BP, Q, mul

H = hashlib.sha256
N_ROWS = 20_000
N_ACCTS = 300
S_UNITS = 200

bp = BP(32, 2)                       # only G and H are used here
G, Hp = bp.G, bp.H


def prf(key, *parts):
    h = H(key)
    for p in parts:
        h.update(struct.pack(">I", len(p)) + p)
    return h.digest()


def det(seed, k, nb=4):
    return int.from_bytes(H(seed + struct.pack(">I", k)).digest()[:nb], "big")


print("=" * 80, flush=True)
print("PAPER B, ITEM 2 -- VERIFIABLE CLASSIFICATION", flush=True)
print("=" * 80, flush=True)

# ------------------------------------------------------------------ ledger
SEED = b"classification-fixed-seed"
K_BLIND = b"amount-blinding-key"
K_CLASS = b"account-blinding-key"        # deliberately NOT K_BLIND

# Zipf-ish account assignment: a few hub accounts carry most rows
weights = [1.0 / (r + 1) ** 1.1 for r in range(N_ACCTS)]
tot_w = sum(weights)
cw, acc = [], 0.0
for w in weights:
    acc += w / tot_w
    cw.append(acc)

accounts, amounts = [], []
for i in range(N_ROWS):
    u = det(SEED + b"acct", i) / 2 ** 32
    accounts.append(bisect.bisect_left(cw, u))
    amounts.append(1 + det(SEED + b"amt", i) % 100_000)

T = sum(amounts)
cum = []
run = 0
for v in amounts:
    run += v
    cum.append(run)

print("  {:,} rows, {} accounts, population total T = {:,}".format(
    N_ROWS, N_ACCTS, T), flush=True)

# account commitments, blinded under a key separate from the amounts'
sigma = [int.from_bytes(prf(K_CLASS, b"sig", struct.pack(">I", i)), "big") % Q
         for i in range(N_ROWS)]
rho = [int.from_bytes(prf(K_BLIND, b"rho", struct.pack(">I", i)), "big") % Q
       for i in range(N_ROWS)]
print("  account blinding derived under a key distinct from the amount "
      "blinding", flush=True)

# pick a target account with enough rows to be worth auditing
by_acct = {}
for i, a in enumerate(accounts):
    by_acct.setdefault(a, []).append(i)
TARGET = max(by_acct, key=lambda a: len(by_acct[a]) if len(by_acct[a]) < 900
             else -1)
rows_a = by_acct[TARGET]
bal_a = sum(amounts[i] for i in rows_a)
print("  auditing account {}: {:,} rows, balance {:,} "
      "({:.2%} of the population)".format(
          TARGET, len(rows_a), bal_a, bal_a / T), flush=True)

# --------------------------------------------- 1. substitution, deterministic
print(flush=True)
print("  1. SUBSTITUTION -- deterministic, closed completely", flush=True)

t0 = time.time()
A = [mul(G, accounts[i]) + mul(Hp, sigma[i]) for i in rows_a]
t_commit = time.time() - t0


def opens_to(commitment, a, s):
    """Anyone can run this: no key, no proof, one point equation."""
    return commitment == mul(G, a) + mul(Hp, s)


honest = all(opens_to(A[k], TARGET, sigma[i]) for k, i in enumerate(rows_a))
print("     every honestly claimed row opens to account {}   : {}".format(
    TARGET, honest), flush=True)

# negative control: the firm swaps in a row belonging to another account,
# keeping the amount identical so the balance stays consistent
other = next(i for i in range(N_ROWS)
             if accounts[i] != TARGET and amounts[i] == amounts[rows_a[0]]) \
    if any(accounts[i] != TARGET and amounts[i] == amounts[rows_a[0]]
           for i in range(N_ROWS)) else \
    next(i for i in range(N_ROWS) if accounts[i] != TARGET)
A_sub = mul(G, accounts[other]) + mul(Hp, sigma[other])
passes = opens_to(A_sub, TARGET, sigma[other])
print("     a row from account {} claimed as account {}      : {}  "
      "(must be False)".format(accounts[other], TARGET, passes), flush=True)

# and the firm cannot invent a sigma' that makes it pass: doing so would open
# one commitment to two values, i.e. solve the discrete log of G to base H
forged = any(opens_to(A_sub, TARGET, (sigma[other] + d) % Q)
             for d in range(1, 2000))
print("     2,000 forged blinding factors tried            : {}  "
      "(must be False)".format(forged), flush=True)
print("     cost: {:.3f} s to build {:,} account commitments, "
      "32 B revealed per claimed row".format(t_commit, len(rows_a)), flush=True)

# ------------------------------------------------- 2. omission, bounded by MUS
print(flush=True)
print("  2. OMISSION -- bounded by monetary-unit sampling", flush=True)
print("     units drawn from the WHOLE ledger; the claimed set tells us "
      "nothing about", flush=True)
print("     what was left out of it", flush=True)
print(flush=True)
print("     {:>8} {:>10} {:>12} {:>16} {:>12}".format(
    "omitted", "M/T", "rows out", "detected (2k trials)", "closed form"),
    flush=True)

TRIALS = 2000
rows_sorted = sorted(rows_a, key=lambda i: -amounts[i])

for frac in (0.0002, 0.001, 0.005, 0.01, 0.02, 0.05):
    target_M = frac * T
    omitted, M = set(), 0
    for i in rows_sorted:
        if M >= target_M:
            break
        omitted.add(i)
        M += amounts[i]
    if not omitted:
        continue

    hits = 0
    for t in range(TRIALS):
        anchor = H(SEED + b"trial" + struct.pack(">I", t)).digest()
        seen = set()
        caught = False
        k = 0
        while len(seen) < S_UNITS:
            u = int.from_bytes(
                H(anchor + b"mus" + struct.pack(">I", k)).digest(), "big") % T
            k += 1
            if u in seen:
                continue
            seen.add(u)
            j = bisect.bisect_right(cum, u)
            if j in omitted:
                caught = True
        hits += caught
    p = hits / TRIALS
    se = 1.96 * (p * (1 - p) / TRIALS) ** 0.5
    closed = 1 - (1 - M / T) ** S_UNITS
    print("     {:>8,} {:>10.4f} {:>12,} {:>10.3f} +/- {:.3f} {:>12.3f}".format(
        M, M / T, len(omitted), p, se, closed), flush=True)

# ---------------------------------------------------------- 3. the swap case
# ------------------------------------------------- 3. the deployment rule
print(flush=True)
print("  3. HOW MANY UNITS DOES AN AUDITOR NEED?", flush=True)
import math
print("     Solving 1-(1-M/T)^s >= 1-beta for s gives "
      "s >= ln(beta)/ln(1-M/T).", flush=True)
print("     Compare with the sample size auditors already compute for MUS: "
      "s = RF/(M/T),", flush=True)
print("     with reliability factor RF = 3.0 at 95% and no expected "
      "misstatement.", flush=True)
print(flush=True)
print("     {:>10} {:>16} {:>16}".format(
    "M/T", "ours (95%)", "classic RF/(M/T)"), flush=True)
for mt in (0.10, 0.05, 0.02, 0.01, 0.005):
    ours = math.ceil(math.log(0.05) / math.log(1 - mt))
    classic = math.ceil(3.0 / mt)
    print("     {:>10.3f} {:>16,} {:>16,}".format(mt, ours, classic),
          flush=True)
print(flush=True)
print("     The two agree to within a unit or two. The cryptographic bound "
      "reduces to", flush=True)
print("     the sample-size rule the profession already uses, so the scheme "
      "needs no new", flush=True)
print("     audit methodology --- only the existing one, applied to "
      "committed amounts.", flush=True)

print(flush=True)
print("  4. WHY THE SPLIT MATTERS", flush=True)
swap_partner = next(i for i in range(N_ROWS) if accounts[i] != TARGET)
print("     A swap moves a row out of account {} and an equal-valued row in."
      .format(TARGET), flush=True)
print("     The balance stays consistent and M is ~0, so the sampling bound "
      "of part 2", flush=True)
print("     gives NO protection: closed form at M/T = 0 is {:.3f}."
      .format(1 - (1 - 0.0) ** S_UNITS), flush=True)
print("     Part 1 rejects it outright. The deterministic check is not a "
      "nicety --- it is", flush=True)
print("     the only thing standing between the auditor and the "
      "zero-magnitude attack.", flush=True)
print(flush=True)
print("done", flush=True)
