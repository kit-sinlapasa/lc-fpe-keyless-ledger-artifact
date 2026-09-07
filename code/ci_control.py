import numpy as np
from harness import World, cluster_ari, ci95
SEEDS=[11,22,33,44,55]
print("CONTROL: is the jump due to seed variance or to refinement depth?", flush=True)
print("  {:>5} {:>8} | {:>22} | per-seed".format("occ","refine","ARI (95% CI)"), flush=True)
for occ in (1.0, 2.0):
    for rf in (1, 3):
        vals=[]
        for s in SEEDS:
            w=World(s, m=300, s=1.0, n_types=400)
            _,_,d,c,_,_,_,_ = w.docs_for_occupancy(occ)
            vals.append(cluster_ari(w, d, c, w.Kp, w.owner, refine=rf))
        m,h,n = ci95(vals)
        print("  {:>5.1f} {:>8} | {:>10.4f} +/- {:.4f} | {}".format(
            occ, rf, m, h, " ".join("{:.3f}".format(v) for v in vals)), flush=True)
