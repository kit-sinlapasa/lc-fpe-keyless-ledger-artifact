"""Why is the structured chart safer? Measure the graph, don't speculate."""
import numpy as np
from collections import Counter
from harness import World
from harness_tfrs import TFRSWorld

for label, mk in (("flat", lambda s: World(s,m=300,s=1.0,n_types=400)),
                  ("TFRS", lambda s: TFRSWorld(s,m=300))):
    w = mk(11)
    da,ca,d_al,c_al,_,_,_,_ = w.docs_for_occupancy(1.0)
    # account-level pairing diversity
    pairs = Counter(zip(da.tolist(), ca.tolist()))
    # alias-level edges: how often does the SAME alias pair recur?
    apairs = Counter(zip(d_al.tolist(), c_al.tolist()))
    rec = Counter(apairs.values())
    # per-alias multiplicity
    mult = Counter(np.concatenate([d_al,c_al]).tolist())
    mv = np.array(list(mult.values()))
    print("{}".format(label), flush=True)
    print("   distinct ACCOUNT pairs used      : {:,}".format(len(pairs)), flush=True)
    print("   distinct ALIAS pairs (edges)     : {:,}".format(len(apairs)), flush=True)
    print("   edges seen once / twice / 3+     : {:,} / {:,} / {:,}".format(
          rec.get(1,0), rec.get(2,0), sum(v for k,v in rec.items() if k>=3)), flush=True)
    print("   alias multiplicity  mean {:.2f}  max {}".format(mv.mean(), mv.max()), flush=True)
    top = np.argsort(-w.f)[:3]
    print("   top-3 accounts: f={} k={}".format(
          ["{:.1%}".format(w.f[i]) for i in top], [int(w.k[i]) for i in top]), flush=True)
    print(flush=True)
