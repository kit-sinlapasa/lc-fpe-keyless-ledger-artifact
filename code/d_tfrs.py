"""Does the occupancy rule survive on a structured TFRS-style chart?"""
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from harness import World, ci95
from harness_tfrs import TFRSWorld

def attack(w,u,v,dims,refine,fast):
    n=w.Kp
    C=sp.coo_matrix((np.ones(2*len(u)),(np.concatenate([u,v]),
      np.concatenate([v,u]))),shape=(n,n)).tocsr()
    C.data=np.log1p(C.data)
    idx=np.nonzero(np.asarray((C!=0).sum(axis=1)).ravel()>0)[0]
    if len(idx) < w.m+2: return float("nan")
    Cs=C[idx][:,idx]
    U,S,_=svds(Cs.asfptype(),k=min(dims,len(idx)-1), random_state=w.seed)
    X=U*S; nn=np.linalg.norm(X,axis=1,keepdims=True); nn[nn==0]=1
    KM=(lambda k: MiniBatchKMeans(k,n_init=3,random_state=w.seed,batch_size=1024)) \
       if fast else (lambda k: KMeans(k,n_init=2,random_state=w.seed))
    lab=KM(w.m).fit_predict(X/nn)
    for _ in range(refine):
        H=sp.coo_matrix((np.ones(len(idx)),(np.arange(len(idx)),lab)),
                        shape=(len(idx),w.m)).tocsr()
        Q=np.asarray((Cs@H).todense())
        nn=np.linalg.norm(Q,axis=1,keepdims=True); nn[nn==0]=1
        lab=KM(w.m).fit_predict(Q/nn)
    return float(adjusted_rand_score(w.owner[idx],lab))

SEEDS=[11,22,33,44,55]
print("="*86, flush=True)
print("D. STRUCTURED TFRS CHART vs FLAT RANDOM PAIRING", flush=True)
print("="*86, flush=True)
w=TFRSWorld(11); print("  sample world: "+w.describe(), flush=True)
w2=World(11, m=300, s=1.0, n_types=400)
print("  flat world  : m={} K'={} max f={:.3%} min f={:.4%}".format(
      w2.m,w2.Kp,float(w2.f.max()),float(w2.f.min())), flush=True)
print(flush=True)
print("  {:<10} {:>5} {:<26} {:>22} {:>8}".format(
      "chart","occ","attack","ARI (95% CI)","upper"), flush=True)
for occ in (0.5, 1.0, 2.0):
    for label,mk in (("flat", lambda s: World(s,m=300,s=1.0,n_types=400)),
                     ("TFRS", lambda s: TFRSWorld(s,m=300))):
        for aname,dims,rf,fast in (("weak  (MB,32,r1)",32,1,True),
                                   ("STRONG(KM,64,r3)",64,3,False)):
            vals=[]; t0=time.time()
            for sd in SEEDS:
                w=mk(sd)
                _,_,d,c,_,_,_,_=w.docs_for_occupancy(occ)
                vals.append(attack(w,d,c,dims,rf,fast))
            m,h,n=ci95(vals); up=m+h
            flag = "" if up<0.10 else "  <== OVER BUDGET"
            print("  {:<10} {:>5.1f} {:<26} {:>10.4f} +/- {:.4f} {:>8.4f}{}  [{:.0f}s]".format(
                  label,occ,aname,m,h,up,flag,time.time()-t0), flush=True)
    print(flush=True)
print("delta (flatness):", flush=True)
for label,mk in (("flat", lambda s: World(s,m=300,s=1.0,n_types=400)),
                 ("TFRS", lambda s: TFRSWorld(s,m=300))):
    m,h,_=ci95([mk(s).delta() for s in SEEDS])
    print("  {:<6} {:.4f} +/- {:.4f}".format(label,m,h), flush=True)
print("done", flush=True)
