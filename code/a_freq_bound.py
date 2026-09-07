#!/usr/bin/env python3
"""
Is a distance-based advantage bound for the frequency channel worth stating?

Theorem 2 bounds Delta, the statistical distance between the aliased class
distribution and uniform. The obvious next theorem is: an adversary that reads
only alias-class frequencies gains at most Delta over guessing. That is true
for one observation. A ledger gives the adversary N of them, and the standard
route from a per-observation distance to an N-observation advantage is either

    TV(p^N, u^N) <= N * TV(p, u)                      (union / subadditivity)
    TV(p^N, u^N) <= sqrt(N * KL(p || u) / 2)          (Pinsker)

and a bound above 1 says nothing at all. So before writing the theorem, this
script computes both at the paper's own operating point, from the allocation
the paper actually uses, and reports whether either is below 1.

This is the check that decides whether the result is worth a page.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H   # noqa: E402


def kl(p, u):
    return sum(pi * math.log(pi / ui) for pi, ui in zip(p, u) if pi > 0)


def tv(p, u):
    return 0.5 * sum(abs(pi - ui) for pi, ui in zip(p, u))


def analyse(m, K, N, seed=0, label=""):
    w = H.World(seed=seed, m=m, K=K)
    f = list(w.f)                   # account frequency profile
    k = [int(x) for x in w.k]       # alias multiplicities from the allocator
    Kp = w.Kp

    # distribution over alias classes: a class of account a carries f(a)/k(a)
    p = []
    for fa, ka in zip(f, k):
        p += [fa / ka] * ka
    tot = sum(p)
    p = [x / tot for x in p]
    u = [1.0 / Kp] * Kp

    d = tv(p, u)
    d_kl = kl(p, u)
    sub = N * d
    pins = math.sqrt(N * d_kl / 2)

    print("  %-22s m=%d K'=%d N=%d" % (label, m, Kp, N), flush=True)
    print("     Delta = TV(p,u)                 : %.6f" % d, flush=True)
    print("     KL(p||u)                        : %.6f" % d_kl, flush=True)
    print("     subadditivity bound  N*Delta    : %.1f   %s"
          % (sub, "USEFUL" if sub < 1 else "vacuous (>= 1)"), flush=True)
    print("     Pinsker bound  sqrt(N*KL/2)     : %.1f   %s"
          % (pins, "USEFUL" if pins < 1 else "vacuous (>= 1)"), flush=True)
    # how few rows would the bound survive?
    n_sub = 1.0 / d if d else float("inf")
    n_pin = 2.0 / d_kl if d_kl else float("inf")
    print("     rows the bounds survive to      : %.0f (subadd), %.0f (Pinsker)"
          % (n_sub, n_pin), flush=True)
    print("", flush=True)
    return d, d_kl, sub, pins


def main():
    print("=" * 74, flush=True)
    print("CAN A DISTANCE BOUND CARRY TO N OBSERVATIONS?", flush=True)
    print("=" * 74, flush=True)
    print("  A ledger period is N rows, so the adversary sees N draws from the", flush=True)
    print("  alias-class distribution, not one. Both standard routes from a", flush=True)
    print("  per-draw distance to an N-draw advantage are computed below.", flush=True)
    print("", flush=True)

    out = []
    for m, K, N in ((300, 10000, 20000), (200, 10000, 20000), (300, 10000, 1000)):
        out.append(analyse(m, K, N, label="operating point" if m == 300 and N == 20000
                                          else "variant"))

    print("=" * 74, flush=True)
    useful = any(s < 1 or p < 1 for _, _, s, p in out)
    if useful:
        print("  At least one bound is below 1: the theorem is worth stating.", flush=True)
    else:
        print("  Both bounds exceed 1 at every point measured. A per-draw distance", flush=True)
        print("  does not carry to a period of this length, so the theorem would", flush=True)
        print("  be vacuous exactly where the paper needs it. This is a reason not", flush=True)
        print("  to write it, and a reason the operating rule is measured instead.", flush=True)
    print("  done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
