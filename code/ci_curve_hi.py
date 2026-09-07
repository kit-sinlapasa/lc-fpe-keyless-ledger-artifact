"""Does the curve really climb to 0.88, or plateau? Extend to high occupancy."""
from harness import World, cluster_ari, ci95
SEEDS=[11,22,33,44,55]
print("dense: m=300, 400 tx types -- HIGH OCCUPANCY", flush=True)
for occ in (32, 64, 128):
    vals=[]
    for sd in SEEDS:
        w=World(sd, m=300, s=1.0, n_types=400)
        _,_,d,c,_,_,_,_ = w.docs_for_occupancy(occ)
        vals.append(cluster_ari(w,d,c,w.Kp,w.owner))
    m,h,n = ci95(vals)
    print("  occ={:>5}  ARI = {:.4f} +/- {:.4f}   pgfplot: ({},{:.4f})".format(
          occ,m,h,occ,m), flush=True)
print("done", flush=True)
