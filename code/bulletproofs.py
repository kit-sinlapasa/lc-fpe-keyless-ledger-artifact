#!/usr/bin/env python3
"""
Aggregated Bulletproofs range proof -- full implementation over NIST P-256.

Follows Bunz et al. (IEEE S&P 2018): Protocol 1 (range proof) composed with
Protocol 2 (inner-product argument), made non-interactive by Fiat-Shamir.

Proves: m Pedersen commitments V_j = v_j*G + gamma_j*H each open to a value in
[0, 2^n).  Proof size is 2*ceil(log2(n*m)) + 4 group elements and 5 scalars,
i.e. 32*(9 + 2k) bytes with k = log2(n*m) -- the formula cited in the paper,
which this implementation checks empirically.
"""
import os
import time
import hashlib
from Crypto.PublicKey import ECC
from Crypto.Math.Numbers import Integer

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import hash_to_curve as _h2c_mod

CURVE = "p256"
# Domain parameters come from hash_to_curve, which writes them out from
# FIPS 186-4 and verifies them (generator on the curve, n*G at infinity),
# rather than from ECC._curves, a private attribute of the arithmetic library.
P = _h2c_mod.P
Q = _h2c_mod.N
B = _h2c_mod.B
G0 = ECC.EccPoint(_h2c_mod.GX, _h2c_mod.GY, curve=CURVE)

# Domain separation tag for this scheme's generators. RFC 9380 section 3.1
# asks for a tag that names the protocol and the suite; ours also names the
# paper so that generators for this construction cannot collide with any
# other use of the same suite.
DST = b"LCFPE-PaperB-BP-v1_P256_XMD:SHA-256_SSWU_RO_"


# --------------------------------------------------------------- group utils
def h2c(label):
    """Generator derivation, RFC 9380 P256_XMD:SHA-256_SSWU_RO_.

    Until 2026-09-07 this was try-and-increment: hash the label, read it as an
    x-coordinate, retry until one lands on the curve. Sound for the purpose --
    a generator needs only an unknown discrete logarithm relative to the
    others -- but not the standard construction, and the paper listed it as a
    departure. h2c_legacy below keeps it so the two can be compared.
    """
    return _h2c_mod.hash_to_curve(label, DST)


def h2c_legacy(label):
    """The try-and-increment derivation used for the measurements recorded
    before 2026-09-07. Kept so that a reader can reproduce those numbers."""
    ctr = 0
    while True:
        d = hashlib.sha256(label + ctr.to_bytes(4, "big")).digest()
        x = int.from_bytes(d, "big") % P
        rhs = (pow(x, 3, P) - 3 * x + B) % P
        y = pow(rhs, (P + 1) // 4, P)
        if pow(y, 2, P) == rhs:
            return ECC.EccPoint(x, y, curve=CURVE)
        ctr += 1


def mul(pt, k):
    k %= Q
    if k == 0:
        return G0.point_at_infinity()
    return pt * Integer(k)


def msm(pts, ks):
    """multi-scalar multiplication (plain; adequate for measurement)"""
    acc = None
    for pt, k in zip(pts, ks):
        k %= Q
        if k == 0:
            continue
        t = pt * Integer(k)
        acc = t if acc is None else acc + t
    return acc if acc is not None else G0.point_at_infinity()


def enc(pt):
    if pt.is_point_at_infinity():
        return b"\x00" * 33
    return bytes([2 + (int(pt.y) & 1)]) + int(pt.x).to_bytes(32, "big")


class TS:
    """Fiat-Shamir transcript"""
    def __init__(self, label):
        self.h = hashlib.sha256(label)

    def add(self, *xs):
        for x in xs:
            self.h.update(x if isinstance(x, bytes)
                          else enc(x) if hasattr(x, "x")
                          else int(x).to_bytes(32, "big"))
        return self

    def chal(self, tag):
        d = hashlib.sha256(self.h.digest() + tag).digest()
        self.h.update(d)
        return int.from_bytes(d, "big") % Q


def ip(a, b):
    return sum(x * y for x, y in zip(a, b)) % Q


def hadamard(a, b):
    return [(x * y) % Q for x, y in zip(a, b)]


# --------------------------------------------------------------- IPA
def ipa_prove(g, h, u, a, b, ts):
    L, R = [], []
    while len(a) > 1:
        k = len(a) // 2
        aL, aR = a[:k], a[k:]
        bL, bR = b[:k], b[k:]
        gL, gR = g[:k], g[k:]
        hL, hR = h[:k], h[k:]
        cL, cR = ip(aL, bR), ip(aR, bL)
        Lk = msm(gR + hL + [u], aL + bR + [cL])
        Rk = msm(gL + hR + [u], aR + bL + [cR])
        L.append(Lk)
        R.append(Rk)
        x = ts.add(Lk, Rk).chal(b"ipa")
        xi = pow(x, Q - 2, Q)
        a = [(aL[i] * x + aR[i] * xi) % Q for i in range(k)]
        b = [(bL[i] * xi + bR[i] * x) % Q for i in range(k)]
        g = [gL[i] * Integer(xi) + gR[i] * Integer(x) for i in range(k)]
        h = [hL[i] * Integer(x) + hR[i] * Integer(xi) for i in range(k)]
    return L, R, a[0], b[0]


def ipa_verify(g, h, u, P_, L, R, a, b, ts):
    for Lk, Rk in zip(L, R):
        x = ts.add(Lk, Rk).chal(b"ipa")
        xi = pow(x, Q - 2, Q)
        k = len(g) // 2
        g = [g[i] * Integer(xi) + g[k + i] * Integer(x) for i in range(k)]
        h = [h[i] * Integer(x) + h[k + i] * Integer(xi) for i in range(k)]
        P_ = mul(Lk, x * x % Q) + P_ + mul(Rk, xi * xi % Q)
    return P_ == mul(g[0], a) + mul(h[0], b) + mul(u, a * b % Q)


# --------------------------------------------------------------- range proof
class BP:
    def __init__(self, n, m):
        self.n, self.m, self.N = n, m, n * m
        self.G = h2c(b"BP/G")
        self.H = h2c(b"BP/H")
        self.U = h2c(b"BP/U")
        self.g = [h2c(b"BP/g/" + str(i).encode()) for i in range(self.N)]
        self.h = [h2c(b"BP/h/" + str(i).encode()) for i in range(self.N)]

    def commit(self, v, gamma):
        return mul(self.G, v) + mul(self.H, gamma)

    def prove(self, vs, gammas, weak=False):
        """weak=True is the prover we shipped first, paired with
        verify(weak=True); see that docstring. Demonstration only."""
        n, m, N = self.n, self.m, self.N
        aL = []
        for v in vs:
            aL += [(v >> b) & 1 for b in range(n)]
        aR = [(x - 1) % Q for x in aL]
        alpha = int.from_bytes(os.urandom(32), "big") % Q
        A = mul(self.H, alpha) + msm(self.g + self.h, aL + aR)
        sL = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(N)]
        sR = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(N)]
        rho = int.from_bytes(os.urandom(32), "big") % Q
        S = mul(self.H, rho) + msm(self.g + self.h, sL + sR)

        # The commitments are public inputs of the statement and MUST be bound
        # into the transcript before any challenge is drawn; omitting them is
        # the weak Fiat-Shamir ("Frozen Heart") flaw, and bp_frozen_heart.py
        # forges a proof against the version that left them out.
        Vs = [self.commit(vs[i], gammas[i]) for i in range(m)]
        ts = TS(b"bp/v1")
        if not weak:
            ts = ts.add(n, m, *Vs)
        ts = ts.add(A, S)
        y = ts.chal(b"y")
        z = ts.chal(b"z")
        yN = [pow(y, i, Q) for i in range(N)]
        z2 = [pow(z, 2 + i // n, Q) * pow(2, i % n, Q) % Q for i in range(N)]

        l0 = [(aL[i] - z) % Q for i in range(N)]
        l1 = sL
        r0 = [(yN[i] * ((aR[i] + z) % Q) + z2[i]) % Q for i in range(N)]
        r1 = hadamard(yN, sR)
        t1 = (ip(l0, r1) + ip(l1, r0)) % Q
        t2 = ip(l1, r1)
        tau1 = int.from_bytes(os.urandom(32), "big") % Q
        tau2 = int.from_bytes(os.urandom(32), "big") % Q
        T1 = mul(self.G, t1) + mul(self.H, tau1)
        T2 = mul(self.G, t2) + mul(self.H, tau2)

        x = ts.add(T1, T2).chal(b"x")
        l = [(l0[i] + l1[i] * x) % Q for i in range(N)]
        r = [(r0[i] + r1[i] * x) % Q for i in range(N)]
        that = ip(l, r)
        taux = (tau2 * x * x + tau1 * x
                + sum(pow(z, 2 + j, Q) * gammas[j] for j in range(m))) % Q
        mu = (alpha + rho * x) % Q

        # rerandomise the inner-product generator with a challenge, so that
        # t-hat is bound to the argument (Bunz et al., Protocol 1)
        w = ts.add(taux, mu, that).chal(b"w")
        Uw = mul(self.U, w)
        yinv = pow(y, Q - 2, Q)
        hp = [self.h[i] * Integer(pow(yinv, i, Q)) for i in range(N)]
        L, R, a_, b_ = ipa_prove(list(self.g), hp, Uw, l, r, ts)
        return dict(A=A, S=S, T1=T1, T2=T2, taux=taux, mu=mu, that=that,
                    L=L, R=R, a=a_, b=b_)

    def verify(self, Vs, pf, weak=False):
        """weak=True reproduces the verifier we shipped first, whose transcript
        omitted the statement (n, m, V) -- the weak Fiat-Shamir flaw. It exists
        only so bp_frozen_heart.py can show the forgery being ACCEPTED by the
        old verifier and REJECTED by this one; never use it for anything."""
        n, m, N = self.n, self.m, self.N
        ts = TS(b"bp/v1")
        if not weak:
            ts = ts.add(n, m, *Vs)
        ts = ts.add(pf["A"], pf["S"])
        y = ts.chal(b"y")
        z = ts.chal(b"z")
        x = ts.add(pf["T1"], pf["T2"]).chal(b"x")
        yN = [pow(y, i, Q) for i in range(N)]
        sum_y = sum(yN) % Q
        delta = ((z - z * z) % Q * sum_y
                 - sum(pow(z, 3 + j, Q) for j in range(m)) * (2 ** n - 1)) % Q
        lhs = mul(self.G, pf["that"]) + mul(self.H, pf["taux"])
        rhs = (msm(Vs, [pow(z, 2 + j, Q) for j in range(m)])
               + mul(self.G, delta) + mul(pf["T1"], x)
               + mul(pf["T2"], x * x % Q))
        if lhs != rhs:
            return False, "t-check"

        w = ts.add(pf["taux"], pf["mu"], pf["that"]).chal(b"w")
        Uw = mul(self.U, w)
        yinv = pow(y, Q - 2, Q)
        hp = [self.h[i] * Integer(pow(yinv, i, Q)) for i in range(N)]
        z2 = [pow(z, 2 + i // n, Q) * pow(2, i % n, Q) % Q for i in range(N)]
        Pp = (pf["A"] + mul(pf["S"], x)
              + msm(self.g, [(-z) % Q] * N)
              + msm(hp, [(z * yN[i] + z2[i]) % Q for i in range(N)])
              + mul(self.H, (-pf["mu"]) % Q)
              + mul(Uw, pf["that"]))      # bind t-hat into the IPA statement
        ok = ipa_verify(list(self.g), hp, Uw, Pp,
                        pf["L"], pf["R"], pf["a"], pf["b"], ts)
        return ok, "ipa" if not ok else "ok"


def size_bytes(pf):
    return 33 * (4 + 2 * len(pf["L"])) + 32 * 5


if __name__ == "__main__":
    import math
    print("=" * 76, flush=True)
    print("AGGREGATED BULLETPROOFS -- real implementation, NIST P-256",
          flush=True)
    print("=" * 76, flush=True)

    n = 64
    print("  {:>4} {:>7} {:>10} {:>10} {:>9} {:>9} {:>8}".format(
        "m", "N=n*m", "prove(s)", "verify(s)", "size(B)", "formula", "B/row"),
        flush=True)
    # The paper extrapolates this curve to the 60,000 values a 20,000-row
    # partition puts under range (C_i, D_i, K_i). An extrapolation is only as
    # good as the span it is fitted over, so the sweep runs as far as the
    # pure-Python prover allows rather than stopping where the table used to:
    # 64 to 16,384 committed bits is 2.4 decades, against 1.2 before.
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        bp = BP(n, m)
        vs = [int.from_bytes(os.urandom(6), "big") % (2 ** 40) for _ in range(m)]
        gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(m)]
        Vs = [bp.commit(vs[j], gs[j]) for j in range(m)]
        t0 = time.time(); pf = bp.prove(vs, gs); tp = time.time() - t0
        t0 = time.time(); ok, why = bp.verify(Vs, pf); tv = time.time() - t0
        k = math.ceil(math.log2(n * m))
        formula = 32 * (9 + 2 * k)
        sz = size_bytes(pf)
        print("  {:>4} {:>7} {:>10.2f} {:>10.2f} {:>9,} {:>9,} {:>8.2f}   {}"
              .format(m, n * m, tp, tv, sz, formula, sz / m,
                      "VERIFIED" if ok else "FAILED:" + why), flush=True)

    print("", flush=True)
    print("  SOUNDNESS TESTS (m=2, n=16)", flush=True)
    bp = BP(16, 2)
    gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(2)]
    Vs = [bp.commit(5, gs[0]), bp.commit(7, gs[1])]
    pf = bp.prove([5, 7], gs)
    print("   honest proof verifies            :", bp.verify(Vs, pf)[0],
          flush=True)

    bad = dict(pf); bad["that"] = (bad["that"] + 1) % Q
    print("   tampered t-hat rejected          :", not bp.verify(Vs, bad)[0],
          flush=True)

    bad2 = dict(pf); bad2["a"] = (bad2["a"] + 1) % Q
    print("   tampered IPA scalar rejected     :", not bp.verify(Vs, bad2)[0],
          flush=True)

    Vs_wrong = [bp.commit(6, gs[0]), bp.commit(7, gs[1])]
    print("   proof bound to its commitments   :",
          not bp.verify(Vs_wrong, pf)[0], flush=True)

    out = 2 ** 16          # out of range for n=16
    gs2 = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(2)]
    Vs2 = [bp.commit(out, gs2[0]), bp.commit(1, gs2[1])]
    pf2 = bp.prove([out, 1], gs2)
    print("   out-of-range value rejected      :", not bp.verify(Vs2, pf2)[0],
          flush=True)
