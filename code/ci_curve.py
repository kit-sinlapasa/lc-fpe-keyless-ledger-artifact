"""Multi-seed ARI-vs-occupancy curves for Fig. 2, both configurations."""
from harness import World, cluster_ari, ci95
SEEDS=[11,22,33,44,55]
CFG=[("dense: m=300, 400 tx types", dict(m=300,s=1.0,n_types=400)),
     ("sparse: m=200, 100 tx types", dict(m=200,s=1.0,n_types=100))]
OCC=[0.5,1,2,4,8,16]
for name,kw in CFG:
    print("="*70, flush=True); print(name, flush=True); print("="*70, flush=True)
    for occ in OCC:
        vals=[]
        for sd in SEEDS:
            w=World(sd, **kw)
            _,_,d,c,_,_,_,_ = w.docs_for_occupancy(occ)
            vals.append(cluster_ari(w,d,c,w.Kp,w.owner))
        m,h,n = ci95(vals)
        print("  occ={:>5.1f}  ARI = {:.4f} +/- {:.4f}   pgfplot: ({},{:.4f})".format(
              occ,m,h,occ,m), flush=True)
print("done", flush=True)
