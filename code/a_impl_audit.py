#!/usr/bin/env python3
"""
Implementation audit of the LC-FPE code layer.

Paper A's FF1 now reproduces the NIST sample values, and the allocation rule has
been stress-tested, but the layer between them -- how a row is turned into a
numeral string and back -- has never been checked. Four properties it has to
have, none of which any existing test exercises:

  1. ROUND TRIP.  The scheme has to be decryptable. FF1 gained a decrypt()
     only recently; nobody has run a row through the whole path and back.

  2. ENCODING INJECTIVITY.  The lifted plaintext is built by string formatting,
     "{:04d}{:03d}{:02d}" over (alias, cost centre, document type). If two
     distinct triples can produce one string, decryption is ambiguous and the
     leakage analysis is wrong. Fixed-width fields make this true only while
     every field stays inside its width.

  3. THE ALIAS FIELD IS FOUR DIGITS.  The code writes alias % 10000. That is an
     undocumented precondition, not a safety net: at K' >= 10^4 two aliases
     silently collide. We check the paper's operating points and then show what
     happens just past them.

  4. ALIAS UNIFORMITY.  FPAE's guarantee is that an account's k(a) aliases each
     carry about the same number of rows; Theorem 2 bounds the residual
     non-uniformity on that basis. The deployed selector is
     HMAC(key, doc:line) mod k(a), and a modulo reduction of a 64-bit value is
     only near-uniform. We measure the realised distribution.
"""
import os
import numpy as np
from collections import Counter
from lcfpe_impl import FF1, alloc_hamilton, alias_index, lift, ALPHA

TWEAK = b"ENT1_2026M09"
FMT = "{:04d}{:03d}{:02d}"


def encode(alias, cc, dt):
    return FMT.format(alias % 10000, cc, dt)


def digits(s):
    return [int(c) for c in s]


print("=" * 82, flush=True)
print("LC-FPE IMPLEMENTATION AUDIT", flush=True)
print("=" * 82, flush=True)

ff1 = FF1(os.urandom(32), radix=10)

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
                    ("one step past the format", 300, 10000)):
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
key = os.urandom(32)
print("     {:>8} {:>10} {:>12} {:>12} {:>10}".format(
    "k(a)", "draws", "mean/alias", "max/mean", "chi2/dof"), flush=True)
for ka in (2, 17, 256, 2508):
    n = 200 * ka
    c = Counter(alias_index(key, i, 0, ka) for i in range(n))
    counts = np.array([c.get(j, 0) for j in range(ka)], dtype=float)
    exp = n / ka
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    print("     {:>8,} {:>10,} {:>12.1f} {:>12.2f} {:>10.3f}".format(
        ka, n, counts.mean(), counts.max() / exp, chi2 / (ka - 1)), flush=True)
print("     chi2/dof near 1.0 means the reduction is not detectably biased",
      flush=True)

print(flush=True)
print("done", flush=True)
