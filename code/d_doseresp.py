"""
Hypothesis from the mechanism measurement: what governs clustering is not
"TFRS vs flat" but PAIRING DIVERSITY -- the number of distinct account pairs
actually posted, which controls how often the same ALIAS pair recurs.

Test it by sweeping within-category concentration on the TFRS chart, which
varies pairing diversity while holding the chart structure fixed.
If the hypothesis holds, ARI should track repeated edges, not chart type.
"""
import numpy as np, scipy.sparse as sp
from collections import Counter
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import World, ci95
from harness_tfrs import TFRSWorld

def attack(w,u,v,dims=64,refine=3):
    n=w.Kp
    C=sp.coo_matrix((np.ones(2*len(u)),(np.concatenate([u,v]),
      np.concatenate([v,u]))),shape=(n,n)).tocsr()
    C.data=np.log1p(C.data)
    idx=np.nonzero(np.asarray((C!=0).sum(axis=1)).ravel()>0)[0]
    Cs=C[idx][:,idx]
    U,S,_=svds(Cs.asfptype(),k=min(dims,len(idx)-1), random_state=w.seed)
    X=U*S; nn=np.linalg.norm(X,axis=1,keepdims=True); nn[nn==0]=1
    KM=lambda k: KMeans(k,n_init=2,random_state=w.seed)
    lab=KM(w.m).fit_predict(X/nn)
    for _ in range(refine):
        H=sp.coo_matrix((np.ones(len(idx)),(np.arange(len(idx)),lab)),
                        shape=(len(idx),w.m)).tocsr()
        Q=np.asarray((Cs@H).todense())
        nn=np.linalg.norm(Q,axis=1,keepdims=True); nn[nn==0]=1
        lab=KM(w.m).fit_predict(Q/nn)
    return float(adjusted_rand_score(w.owner[idx],lab))

def stats(w, occ=1.0):
    da,ca,d_al,c_al,_,_,_,_ = w.docs_for_occupancy(occ)
    ap=Counter(zip(d_al.tolist(),c_al.tolist()))
    rep=sum(1 for v in ap.values() if v>=2)
    return len(Counter(zip(da.tolist(),ca.tolist()))), rep, attack(w,d_al,c_al)

SEEDS=[11,22,33]
print("="*88, flush=True)
print("DOSE-RESPONSE: pairing diversity -> repeated alias edges -> ARI  (occ=1)", flush=True)
print("="*88, flush=True)
print("  {:<28} {:>14} {:>14} {:>20}".format(
      "world","acct pairs","repeated edges","ARI (95% CI)"), flush=True)
rows=[]
for label, mk in (
    ("TFRS cat_skew=0.3", lambda s: TFRSWorld(s,m=300,cat_skew=0.3)),
    ("TFRS cat_skew=1.0", lambda s: TFRSWorld(s,m=300,cat_skew=1.0)),
    ("TFRS cat_skew=1.8", lambda s: TFRSWorld(s,m=300,cat_skew=1.8)),
    ("TFRS cat_skew=2.6", lambda s: TFRSWorld(s,m=300,cat_skew=2.6)),
    ("flat 1600 tx types", lambda s: World(s,m=300,s=1.0,n_types=1600)),
    ("flat  400 tx types", lambda s: World(s,m=300,s=1.0,n_types=400)),
    ("flat  100 tx types", lambda s: World(s,m=300,s=1.0,n_types=100)),
):
    P,R,A=[],[],[]
    for sd in SEEDS:
        p,r,a = stats(mk(sd)); P.append(p); R.append(r); A.append(a)
    am,ah,_=ci95(A)
    rows.append((np.mean(R), am))
    print("  {:<28} {:>14,.0f} {:>14,.0f} {:>12.4f} +/- {:.4f}".format(
          label, np.mean(P), np.mean(R), am, ah), flush=True)
r=np.array([x[0] for x in rows]); a=np.array([x[1] for x in rows])
print(flush=True)
print("  Pearson corr(repeated edges, ARI) = {:+.3f}".format(
      float(np.corrcoef(r,a)[0,1])), flush=True)
print("  Pearson corr(log repeated, ARI)   = {:+.3f}".format(
      float(np.corrcoef(np.log(r+1),a)[0,1])), flush=True)
