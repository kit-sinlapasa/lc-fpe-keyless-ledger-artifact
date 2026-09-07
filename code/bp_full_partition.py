#!/usr/bin/env python3
"""
Cost-scaling ladder for one partition-level range-proof batch at n=32.

Table tab:bp stops at n*m = 16,384 and extrapolates from there to the whole
partition. This script closes that gap by measurement instead. It runs a
ladder of sizes and writes every rung as soon as it finishes, so a rung that
runs out of memory still leaves every rung below it measured.

The deployed verifier is mixed-width. Its 20,000 C_i values use n_C=64 and
pad to m=32,768; its 40,000 D_i,K_i values use n_S=32 and pad to m=65,536.
Both batches therefore have n*m=2,097,152, and the full verifier runs both.
This n=32 ladder measures the cost of one batch; it is not by itself an
execution of the full verifier.

MEMORY MODEL, and why it is the fitted one rather than the first one.
The first version of this script predicted a rung's footprint from a
micro-benchmark that timed and sized `BP.__init__` on its own: 420 bytes per
curve point, times 1.9 for prover transients. Against the rungs this script
actually ran, that was wrong by a factor of three and a half -- it predicted
380 MB for n*m = 131,072, which used 106 MB. A guard built on it would have
parked the ladder at n*m = 262,144 and reported a gap eight times larger than
the machine could really have closed. The model below is a straight-line fit
to the peak RSS of the rungs measured here:

    n*m =  32,768  ->   69 MB          fit: 56.7 MB + 0.000376 MB per n*m
    n*m = 131,072  ->  106 MB               (about 197 bytes per generator)

A micro-benchmark of one allocation is not a measurement of the process that
does the work. Where the two disagree, the process wins.

A rung that does not fit is not skipped and not forced into the page file:
the ladder waits, re-checking every minute, because free memory on a desktop
is a function of what else is open. A time measured while paging measures the
page file, not the prover.
"""
import argparse
import gc
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bulletproofs import BP, Q, size_bytes   # noqa: E402

try:
    import psutil
    _P = psutil.Process()
except Exception:
    psutil = None
    _P = None

N_BITS = 32                  # side-proof width and cost-control width
REAL_VALUES = 40_000         # 20,000 rows x (D_i, K_i) in this batch
LADDER = (1024, 4096, 8192, 16384, 32768, 65536)

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = (os.path.dirname(HERE) if os.path.basename(HERE) == "code"
            else os.path.join(HERE, "..", "artifact"))
RESULT_FILE = os.path.join(ARTIFACT, "results", "bp_full_partition.txt")
FALLBACK_B_PER_POINT = 500   # used only until two rungs have been measured
MEM_SAFETY = 1.35
CUSHION_MB = 250             # never drive the machine to zero
WAIT_HOURS = 6.0
POLL_S = 60


def avail_mb():
    return None if psutil is None else psutil.virtual_memory().available / 2 ** 20


def rss_mb():
    """The true high-water mark, not the reading after the prover has finished.

    The first version of this sampled rss once prove() returned, which is
    after the inner-product argument has released its transients, and it
    understated badly: the n*m=524,288 rung was recorded at a level the live
    process exceeded twofold while still proving. Windows keeps the real
    figure in peak_wset, so use it where it exists and say so.
    """
    if not _P:
        return float("nan")
    mi = _P.memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2 ** 20


def measured_peaks(path=RESULT_FILE):
    """(n*m, peak RSS MB) for every rung already in the result file."""
    out = []
    try:
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*([\d,]+)\s+([\d,]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+"
                         r"\s+[\d,]+\s+yes\s+(\d+)M", line)
            if m:
                out.append((int(m.group(2).replace(",", "")), float(m.group(3))))
    except OSError:
        pass
    return sorted(set(out))


def need_mb(N):
    """Predict a rung's footprint from the rungs this machine actually ran.

    Fitting a constant once and trusting it is what went wrong twice here. A
    micro-benchmark of BP.__init__ said 420 B per point; the first two rungs
    implied 197 and the guard became far too generous; the third rung implied
    468 and the same guard became too tight, which is the dangerous direction,
    because a rung that starts without room measures the page file rather than
    the prover. So the slope is refitted from the two largest rungs measured so
    far, every time, and it self-corrects as the ladder climbs.
    """
    pts = measured_peaks()
    if len(pts) >= 2:
        (N1, M1), (N2, M2) = pts[-2], pts[-1]
        if N2 > N1:
            b = (M2 - M1) / (N2 - N1)
            a = M2 - b * N2
            return max(a + b * N, M2) * MEM_SAFETY + CUSHION_MB
    return 2 * N * FALLBACK_B_PER_POINT / 2 ** 20 * MEM_SAFETY + CUSHION_MB


def wait_for(N, m):
    need = need_mb(N)
    free = avail_mb()
    if free is None or free >= need:
        return True
    print("  m=%d needs about %.0f MB; %.0f MB free. Waiting up to %.0f h"
          " (closing other applications releases it)."
          % (m, need, free, WAIT_HOURS), flush=True)
    deadline = time.time() + WAIT_HOURS * 3600
    while time.time() < deadline:
        time.sleep(POLL_S)
        free = avail_mb()
        if free >= need:
            print("  m=%d: %.0f MB free now, starting." % (m, free), flush=True)
            return True
    print("  m=%d: still %.0f MB free after %.0f h, need %.0f MB."
          % (m, free, WAIT_HOURS, need), flush=True)
    return False


def run(n, m):
    t0 = time.time()
    bp = BP(n, m)
    t_setup = time.time() - t0

    vs = [int.from_bytes(os.urandom(4), "big") % (2 ** (n - 1)) for _ in range(m)]
    gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(m)]
    Vs = [bp.commit(vs[j], gs[j]) for j in range(m)]

    t0 = time.time(); pf = bp.prove(vs, gs); t_prove = time.time() - t0
    peak = rss_mb()
    t0 = time.time(); ok, why = bp.verify(Vs, pf); t_verify = time.time() - t0
    sz = size_bytes(pf)
    del bp, vs, gs, Vs, pf
    gc.collect()
    return t_setup, t_prove, t_verify, sz, ok, why, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0,
                    help="skip rungs below this m (resume a ladder)")
    a = ap.parse_args()
    rungs = [m for m in LADDER if m >= a.start]

    if not a.start:
        print("=" * 78, flush=True)
        print("PARTITION-BATCH COST LADDER -- n=%d" % N_BITS, flush=True)
        print("=" * 78, flush=True)
        print("  %d real values (20,000 rows x D_i, K_i). n*m must be a power"
              % REAL_VALUES, flush=True)
        print("  of two, so the top rung pads to m=65,536 and its cost is an upper",
              flush=True)
        print("  bound for this batch. The separate C_i batch has the same n*m.",
              flush=True)
    else:
        print("", flush=True)
        print("  RESUMED at m=%d with the memory model refitted to the rungs above"
              % a.start, flush=True)
    print("  memory guard: refit from measured peak RSS, x%.2f, %d MB cushion."
          % (MEM_SAFETY, CUSHION_MB), flush=True)
    print("  available now: %s MB."
          % ("?" if avail_mb() is None else "%.0f" % avail_mb()), flush=True)
    print("", flush=True)
    print("  {:>7} {:>10} {:>9} {:>11} {:>11} {:>9} {:>8} {:>9}".format(
        "m", "n*m", "setup(s)", "prove(s)", "verify(s)", "size(B)", "sound?",
        "peakRSS"), flush=True)

    prev = None
    reached = 0
    for m in rungs:
        N = N_BITS * m
        if not wait_for(N, m):
            print("", flush=True)
            print("  Ladder stops before m=%d." % m, flush=True)
            break
        try:
            ts, tp, tv, sz, ok, why, peak = run(N_BITS, m)
        except MemoryError:
            print("", flush=True)
            print("  MemoryError at m=%d despite the check; ladder stops." % m,
                  flush=True)
            break
        k = math.ceil(math.log2(N))
        print("  {:>7,} {:>10,} {:>9.1f} {:>11.1f} {:>11.1f} {:>9,} {:>8} {:>8.0f}M"
              .format(m, N, ts, tp, tv, sz, "yes" if ok else "NO:" + why, peak),
              flush=True)
        if prev:
            pm, pp, pv = prev
            print("       -> vs m={:,}: n*m x{:.0f}, prove x{:.2f}, verify x{:.2f},"
                  " k={}".format(pm, N / (N_BITS * pm), tp / pp, tv / pv, k),
                  flush=True)
        prev = (m, tp, tv)
        reached = m
        if m >= 65536:
            print("", flush=True)
            print("  This rung covers the side-proof batch: %d committed values, of"
                  % m, flush=True)
            print("  which %d are real. The full verifier still requires the C_i batch."
                  % REAL_VALUES, flush=True)

    print("", flush=True)
    print("  reached m=%d (n*m=%d)" % (reached, N_BITS * reached), flush=True)
    print("  done", flush=True)


if __name__ == "__main__":
    main()
