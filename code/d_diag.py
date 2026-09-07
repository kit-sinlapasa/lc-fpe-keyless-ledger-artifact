"""Is the TFRS advantage real, or an artifact of unreachable accounts?"""
import numpy as np
from harness import World
from harness_tfrs import TFRSWorld

for label, mk in (("flat", lambda s: World(s,m=300,s=1.0,n_types=400)),
                  ("TFRS", lambda s: TFRSWorld(s,m=300))):
    w = mk(11)
    nz = int((w.f > 0).sum())
    print("{}: m={}  accounts with f>0 = {}  ({} DEAD)".format(
          label, w.m, nz, w.m-nz), flush=True)
    if label=="TFRS":
        from collections import Counter
        c = Counter(w.cat[i] for i in range(w.m) if w.f[i]==0)
        print("   dead by category:", dict(c), flush=True)
    for occ in (1.0, 2.0):
        _,_,d,c_,_,_,_,_ = w.docs_for_occupancy(occ)
        al = np.concatenate([d,c_])
        seen_al = len(np.unique(al))
        seen_ac = len(np.unique(w.owner[np.unique(al)]))
        print("   occ={:.1f}: aliases seen {}/{} ({:.0%})   accounts seen {}/{}"
              "   asked for {} clusters".format(
              occ, seen_al, w.Kp, seen_al/w.Kp, seen_ac, w.m, w.m), flush=True)
    print("   top-10 k(a):", sorted(w.k, reverse=True)[:10], flush=True)
    print(flush=True)
