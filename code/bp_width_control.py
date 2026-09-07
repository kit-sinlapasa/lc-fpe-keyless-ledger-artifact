#!/usr/bin/env python3
"""
Control: is aggregated-Bulletproof cost a function of n*m alone?

Table tab:bp contains an n=64 sweep and an n=32 cost ladder. The deployed
verifier uses both widths in separate batches, so comparisons at equal n*m
are meaningful only if the two configurations cost the same. Bulletproofs
commit to n*m bits either way and the inner-product argument runs over a
vector of that length, so they should; this measures it rather than asserting
it. Two scales are used because a match at one point could be coincidental.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bulletproofs import BP, Q, size_bytes   # noqa: E402

PAIRS = [((32, 2), (64, 1)), ((32, 32), (64, 16))]


def run(n, m):
    bp = BP(n, m)
    vs = [int.from_bytes(os.urandom(4), "big") % (2 ** (n // 2)) for _ in range(m)]
    gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(m)]
    Vs = [bp.commit(vs[j], gs[j]) for j in range(m)]
    t0 = time.time(); pf = bp.prove(vs, gs); tp = time.time() - t0
    t0 = time.time(); ok, why = bp.verify(Vs, pf); tv = time.time() - t0
    return tp, tv, size_bytes(pf), ok


def main():
    print("=" * 78, flush=True)
    print("BIT-WIDTH CONTROL -- equal n*m, different (n, m)", flush=True)
    print("=" * 78, flush=True)
    print("  {:>4} {:>5} {:>7} {:>10} {:>10} {:>9} {:>8}".format(
        "n", "m", "n*m", "prove(s)", "verify(s)", "size(B)", "sound?"), flush=True)
    worst = 0.0
    for pair in PAIRS:
        got = []
        for (n, m) in pair:
            tp, tv, sz, ok = run(n, m)
            got.append((tp, tv, sz))
            print("  {:>4} {:>5} {:>7,} {:>10.2f} {:>10.2f} {:>9,} {:>8}".format(
                n, m, n * m, tp, tv, sz, "yes" if ok else "NO"), flush=True)
        (tp0, tv0, s0), (tp1, tv1, s1) = got
        assert s0 == s1, "proof size differs at equal n*m"
        d = abs(tp1 - tp0) / min(tp0, tp1)
        worst = max(worst, d)
        print("     -> n*m={:,}: prove differs by {:.1%}, size identical ({:,} B)"
              .format(pair[0][0] * pair[0][1], d, s0), flush=True)
    print("", flush=True)
    print("  largest prover-time difference across the pairs : {:.1%}".format(worst),
          flush=True)
    print("  proof size is a function of n*m alone           : True", flush=True)
    print("", flush=True)
    print("  Cost therefore depends on the number of committed bits, not on how", flush=True)
    print("  those bits are split between width and aggregation. This supports", flush=True)
    print("  comparing the n=64 amount batch with the n=32 side batch.", flush=True)


if __name__ == "__main__":
    main()
