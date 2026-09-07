#!/usr/bin/env python3
"""
Offset-encoded Pedersen amounts -- correctness + re-measurement.

Problem found by the baseline run: credits are negative, so v mod n is a full
256-bit scalar and the m*G multiplication costs 2.2x what a small scalar does.

Fix: commit to  v + OFFSET  (OFFSET = 2^40) so the scalar is always positive
and ~41 bits.  Thai satang amounts up to ~1e11 (1 billion baht) still fit
below OFFSET, so v + OFFSET stays in [0, 2^41).

The keyless period-close check survives the change:
    sum_i C_i  =  (sum_i v_i + N*OFFSET) * G  +  R*H
and for a balanced ledger sum_i v_i = 0, so the auditor checks
    sum_i C_i  ==  (N*OFFSET)*G + R*H
using only N (which is public in the leakage profile) and the opened R.
"""
import os
import time
import hashlib
import numpy as np
from Crypto.PublicKey import ECC
from Crypto.Math.Numbers import Integer
import lcfpe_impl as L

CURVE = "p256"
_c = ECC._curves[CURVE]
P, ORDER, Bc = int(_c.p), int(_c.order), int(_c.b)
G = ECC.EccPoint(int(_c.Gx), int(_c.Gy), curve=CURVE)
OFFSET = 1 << 40


def hash_to_point(label):
    ctr = 0
    while True:
        h = hashlib.sha256(label + ctr.to_bytes(4, "big")).digest()
        x = int.from_bytes(h, "big") % P
        rhs = (pow(x, 3, P) - 3 * x + Bc) % P
        y = pow(rhs, (P + 1) // 4, P)
        if pow(y, 2, P) == rhs:
            return ECC.EccPoint(x, y, curve=CURVE)
        ctr += 1


H = hash_to_point(b"LC-FPE/pedersen/H/v1")

print("=" * 80, flush=True)
print("OFFSET-ENCODED PEDERSEN -- correctness and re-measurement", flush=True)
print("=" * 80, flush=True)
print("  OFFSET = 2^40 = {:,}".format(OFFSET), flush=True)

# ------------------------------------------------------------- correctness
rng = np.random.default_rng(5)
n_t = 400
v = rng.integers(-10 ** 7, 10 ** 7, n_t).astype(object)
v[-1] = -int(sum(int(x) for x in v[:-1]))          # balanced
r = [int.from_bytes(os.urandom(32), "big") % ORDER for _ in range(n_t)]

cs = [(G * Integer((int(v[i]) + OFFSET) % ORDER)) + (H * Integer(r[i]))
      for i in range(n_t)]
acc = cs[0].copy()
for c in cs[1:]:
    acc += c
R = sum(r) % ORDER
target = (G * Integer((n_t * OFFSET) % ORDER)) + (H * Integer(R))
ok = int(acc.x) == int(target.x) and int(acc.y) == int(target.y)
print("  balanced ledger, keyless check passes : {}".format(ok), flush=True)

cs_bad = list(cs)
cs_bad[7] = (G * Integer((int(v[7]) + 1 + OFFSET) % ORDER)) + (H * Integer(r[7]))
acc_b = cs_bad[0].copy()
for c in cs_bad[1:]:
    acc_b += c
print("  1-satang tamper detected              : {}"
      .format(int(acc_b.x) != int(target.x)), flush=True)

# ------------------------------------------------------------- speed
S = 1500
amt = rng.integers(-10 ** 7, 10 ** 7, S)
pool = [H * Integer(int.from_bytes(os.urandom(32), "big") % ORDER)
        for _ in range(256)]

t0 = time.time()
for i in range(S):
    (G * Integer(int(amt[i]) % ORDER)) + pool[i % 256]
t_old = (time.time() - t0) / S

t0 = time.time()
for i in range(S):
    (G * Integer(int(amt[i]) + OFFSET)) + pool[i % 256]
t_new = (time.time() - t0) / S

print("", flush=True)
print("  Pedersen commit, signed  (v mod n) : {:7.1f} us/row".format(t_old * 1e6),
      flush=True)
print("  Pedersen commit, offset  (v+2^40)  : {:7.1f} us/row   {:.1f}x faster"
      .format(t_new * 1e6, t_old / t_new), flush=True)

# ------------------------------------------------------------- end-to-end
M = 200
K = 10000 - M
w = 1.0 / np.arange(1, M + 1)
f = w / w.sum()
k = L.alloc_hamilton(f, K)
startv = np.concatenate([[0], np.cumsum(k)[:-1]])
ff1 = L.FF1(os.urandom(32), radix=10)
ka = os.urandom(32)

print("", flush=True)
print("  END-TO-END LC-FPE per row (3 seeds, 800 rows each)", flush=True)
tot = []
for seed in (1, 2, 3):
    rg = np.random.default_rng(seed)
    a_ = rg.choice(M, size=800, p=f)
    c_ = rg.integers(0, 100, 800)
    d_ = rg.integers(0, 10, 800)
    m_ = rg.integers(-10 ** 7, 10 ** 7, 800)
    t0 = time.time()
    for i in range(800):
        a = int(a_[i])
        al = startv[a] + L.alias_index(ka, i, 0, int(k[a]))
        s = "{:04d}{:03d}{:02d}".format(int(al) % 10000, int(c_[i]), int(d_[i]))
        ff1.encrypt([int(ch) for ch in s], tweak=b"ENT1_2026M09")
        (G * Integer(int(m_[i]) + OFFSET)) + pool[i % 256]
    tot.append((time.time() - t0) / 800)
mu, sd = float(np.mean(tot)) * 1e6, float(np.std(tot)) * 1e6
print("    {:.0f} +/- {:.0f} us/row   ->  {:,.0f} rows/s   "
      "({:.1f}s for 20,000 rows)".format(mu, sd, 1e6 / mu, mu * 20000 / 1e6),
      flush=True)
print("", flush=True)
print("  was 749 us/row with signed amounts  ->  now {:.0f} us/row  "
      "= {:.1f}x faster end-to-end".format(mu, 749 / mu), flush=True)
