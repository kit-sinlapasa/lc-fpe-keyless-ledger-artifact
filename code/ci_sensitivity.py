"""Re-run the 14-config sensitivity sweep with 5 seeds and 95% CIs."""
import numpy as np
from harness import World, cluster_ari, ci95
SEEDS=[11,22,33,44,55]

def run(m, s, nt, occ):
    vals=[]
    for sd in SEEDS:
        w = World(sd, m=m, s=s, n_types=nt)
        _,_,d,c,_,_,_,_ = w.docs_for_occupancy(occ)
        vals.append(cluster_ari(w, d, c, w.Kp, w.owner))
    return ci95(vals)

def dlt(m, s, nt):
    return ci95([World(sd, m=m, s=s, n_types=nt).delta() for sd in SEEDS])

print("="*94, flush=True)
print("SENSITIVITY SWEEP -- 5 seeds, 95% CI   (alpha=1e4, K=alpha-m)", flush=True)
print("="*94, flush=True)
print("  {:<18} {:>17} | {:>19} | {:>19}".format(
      "setting","Delta (CI)","ARI occ=1 (CI)","ARI occ=2 (CI)"), flush=True)

def row(label, m, s, nt):
    dm,dh,_ = dlt(m,s,nt)
    a1,h1,_ = run(m,s,nt,1.0)
    a2,h2,_ = run(m,s,nt,2.0)
    hi1 = a1+h1
    reg = "safe" if hi1 < 0.10 else "LEAKING"
    print("  {:<18} {:>8.4f}+/-{:.4f} | {:>10.4f}+/-{:.4f} | {:>10.4f}+/-{:.4f}  {}".format(
          label, dm,dh, a1,h1, a2,h2, reg), flush=True)

print("  --- chart size m ---", flush=True)
for m in (50,100,200,400): row("m={}".format(m), m, 1.0, 400)
print("  --- skew s ---", flush=True)
for s in (0.6,0.8,1.0,1.3,1.6): row("s={}".format(s), 200, s, 400)
print("  --- pairing sparsity ---", flush=True)
for nt in (100,200,400,800,1600): row("types={}".format(nt), 200, 1.0, nt)
print("\ndone", flush=True)
