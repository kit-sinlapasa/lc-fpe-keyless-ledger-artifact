#!/usr/bin/env python3
"""
Implementation audit of the LC-FPE code layer.

Paper A's FF1 reproduces the NIST sample values and the allocation rule has
been stress-tested. This audit checks the layer between those components and
the byte-level/key-hierarchy contract shared with Paper B:

  1. ROUND TRIP.  The scheme has to be decryptable. FF1 gained a decrypt()
     only recently; nobody has run a row through the whole path and back.

  2. ENCODING INJECTIVITY.  The lifted plaintext is built by string formatting,
     "{:04d}{:03d}{:02d}" over (alias, cost centre, document type). If two
     distinct triples can produce one string, decryption is ambiguous and the
     leakage analysis is wrong. Fixed-width fields make this true only while
     every field stays inside its width.

  3. THE ALIAS FIELD IS FOUR DIGITS.  The old code wrote alias % 10000. That
     was an undocumented precondition, not a safety net: at K' > 10^4 aliases
     silently collide. We check the paper's operating points and then show what
     happens just past them.

  4. ALIAS UNIFORMITY.  FPAE's guarantee is that an account's k(a) aliases each
     carry about the same number of rows; Theorem 2 bounds the residual
     non-uniformity on that basis. The deployed selector uses domain-separated
     HMAC output and exact rejection sampling. We measure the realised
     distribution.

  5. KEY AND ROWID CONTRACT. HKDF labels must separate the component keys and
     the shared RowID must be a reversible 24-byte fixed-width encoding.
"""
import numpy as np
from collections import Counter
from scipy.stats import chi2 as chi2_dist
from lcfpe_impl import (
    ALIAS_SECURITY_BITS, MAX_ALIAS_BLOCKS, FF1, ALPHA, alias_index,
    alloc_hamilton, derive_key, lift, row_id,
)
from paperb_protocol import ROW_ID_BYTES, decode_row_id

TWEAK = b"ENT1_2026M09"
FMT = "{:04d}{:03d}{:02d}"


def encode(alias, cc, dt):
    return FMT.format(alias % 10000, cc, dt)


def digits(s):
    return [int(c) for c in s]


print("=" * 82, flush=True)
print("LC-FPE IMPLEMENTATION AUDIT", flush=True)
print("=" * 82, flush=True)

ff1 = FF1(bytes(range(32)), radix=10)

# ---------------------------------------------------------------- 1. round trip
print(flush=True)
print("  1. ROUND TRIP through the lifted domain", flush=True)
rng = np.random.default_rng(11)
bad = 0
for _ in range(2000):
    al = int(rng.integers(0, 9800))
    cc = int(rng.integers(0, 100))
    dt = int(rng.integers(0, 10))
    s = encode(al, cc, dt)
    ct = ff1.encrypt(digits(s), tweak=TWEAK)
    back = "".join(str(d) for d in ff1.decrypt(ct, tweak=TWEAK))
    if back != s:
        bad += 1
    else:
        al2, cc2, dt2 = int(back[:4]), int(back[4:7]), int(back[7:])
        if (al2, cc2, dt2) != (al, cc, dt):
            bad += 1
print("     2,000 rows encrypted and recovered exactly : {}".format(bad == 0),
      flush=True)
print("     ciphertext is 9 digits, same as plaintext  : {}".format(
    len(ff1.encrypt(digits(encode(1, 2, 3)), tweak=TWEAK)) == 9), flush=True)

# ---------------------------------------------------- 2. encoding injectivity
print(flush=True)
print("  2. ENCODING INJECTIVITY over the declared ranges", flush=True)
seen = {}
clash = None
for al in range(0, 9800, 7):
    for cc in range(0, 100, 3):
        for dt in range(10):
            s = encode(al, cc, dt)
            if s in seen and seen[s] != (al, cc, dt):
                clash = (seen[s], (al, cc, dt), s)
                break
            seen[s] = (al, cc, dt)
print("     {:,} triples, all encodings distinct       : {}".format(
    len(seen), clash is None), flush=True)

# ------------------------------------------------- 3. the four-digit precondition
print(flush=True)
print("  3. THE ALIAS FIELD HOLDS FOUR DIGITS", flush=True)
print("     {:<34} {:>8} {:>10}".format("operating point", "K'", "safe?"),
      flush=True)
for label, m, K in (("tab:ari, tab:tfrs  m=300", 300, 9700),
                    ("tab:ablation       m=200", 200, 9800),
                    ("tab:sens alpha=1e4 m=400", 400, 9600),
                    ("one step past the format", 300, 10001)):
    w = 1.0 / np.arange(1, m + 1)
    f = w / w.sum()
    Kp = int(alloc_hamilton(f, K).sum())
    print("     {:<34} {:>8,} {:>10}".format(
        label, Kp, "yes" if Kp <= 10000 else "NO -- COLLIDES"), flush=True)
print(flush=True)
print("     the wrapping encoder collides silently: alias 3 and alias 10003 "
      "both give {}".format(encode(10003, 0, 0)), flush=True)
try:
    lift(ALPHA, 0, 0)
    guard = "NO GUARD -- still wraps"
except ValueError:
    guard = "raises, as Setup specifies"
print("     lift() at alias = alpha = {:,}: {}".format(ALPHA, guard), flush=True)
print("     lift() inside the field returns {}".format(lift(9999, 99, 9)),
      flush=True)

# ------------------------------------------------------- 4. alias uniformity
print(flush=True)
print("  4. ALIAS UNIFORMITY of rejection-sampled PRF(key, RowID) on [0, k(a))", flush=True)
key = bytes(reversed(range(32)))
print("     {:>8} {:>10} {:>12} {:>12} {:>10} {:>10}".format(
    "k(a)", "draws", "mean/alias", "max/mean", "chi2/dof", "p-value"),
    flush=True)
p_values = []
for ka in (2, 17, 256, 2508):
    n = 200 * ka
    c = Counter(alias_index(
        key, i, 0, ka, rid=row_id(1, 202609, 0, 0, i, 0))
        for i in range(n))
    counts = np.array([c.get(j, 0) for j in range(ka)], dtype=float)
    exp = n / ka
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p_value = float(chi2_dist.sf(chi2, ka - 1))
    p_values.append(p_value)
    print("     {:>8,} {:>10,} {:>12.1f} {:>12.2f} {:>10.3f} {:>10.3f}".format(
        ka, n, counts.mean(), counts.max() / exp, chi2 / (ka - 1), p_value),
        flush=True)
holm_rejects = False
for rank, p_value in enumerate(sorted(p_values)):
    if p_value > 0.05 / (len(p_values) - rank):
        break
    holm_rejects = True
print("     no chi-square test rejects after Holm correction : {}".format(
    not holm_rejects), flush=True)
print("     rejection cap meets {:d}-bit failure target       : {}".format(
    ALIAS_SECURITY_BITS,
    18 * MAX_ALIAS_BLOCKS >= ALIAS_SECURITY_BITS), flush=True)

# ---------------------------------------------------- 5. key/RowID contract
print(flush=True)
print("  5. KEY SEPARATION AND THE SHARED ROWID CONTRACT", flush=True)
master = bytes(range(32))
labels = (b"fpe", b"alias", b"blind", b"aead")
derived = [derive_key(master, label) for label in labels]
repeat = [derive_key(master, label) for label in labels]
print("     HKDF derivation is deterministic             : {}".format(
    derived == repeat), flush=True)
print("     four component labels give distinct keys     : {}".format(
    len(set(derived)) == len(derived)), flush=True)
rid_fields = (17, 202609, 3, 4, 5_000_000_007, 19)
rid = row_id(*rid_fields)
print("     RowID is exactly {:d} bytes                   : {}".format(
    ROW_ID_BYTES, len(rid) == ROW_ID_BYTES), flush=True)
print("     RowID fields round-trip exactly               : {}".format(
    decode_row_id(rid) == rid_fields), flush=True)

print(flush=True)
print("done", flush=True)
