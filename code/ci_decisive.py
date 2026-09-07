"""
DECISIVE TEST.  The design rule rests on ARI at occupancy 1 (and 2).  Those
numbers were produced by the harness attack, which the occupancy-103 control
just showed to be ~0.13 ARI weaker than the stronger configuration.  If the
stronger adversary pushes occupancy 1 past the 0.10 budget, the rule fails.
"""
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from harness import World, ci95

def attack(w,u,v,dims,refine,fast):
    n=w.Kp
    C=sp.coo_matrix((np.ones(2*len(u)),(np.concatenate([u,v]),
      np.concatenate([v,u]))),shape=(n,n)).tocsr()
    C.data=np.log1p(C.data)
    idx=np.nonzero(np.asarray((C!=0).sum(axis=1)).ravel()>0)[0]
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
CFGS=[("weak  (MiniBatch,32,r1)",32,1,True),
      ("STRONG(full KM,  64,r3)",64,3,False)]
print("DECISIVE: does the stronger adversary break the occupancy rule?", flush=True)
print("  {:>4} {:<26} {:>22}  upper  verdict".format("occ","attack","ARI (95% CI)"), flush=True)
for occ in (1.0, 2.0):
    for label,dims,refine,fast in CFGS:
        vals=[]; t0=time.time()
        for sd in SEEDS:
            w=World(sd,m=300,s=1.0,n_types=400)
            _,_,d,c,_,_,_,_=w.docs_for_occupancy(occ)
            vals.append(attack(w,d,c,dims,refine,fast))
        m,h,n=ci95(vals); up=m+h
        print("  {:>4.1f} {:<26} {:>10.4f} +/- {:.4f}  {:.4f}  {}   [{:.0f}s]".format(
            occ,label,m,h,up,"SAFE" if up<0.10 else "*** BUDGET EXCEEDED ***",
            time.time()-t0), flush=True)
print("done", flush=True)
