"""
Calibrate the rule  occupancy <= C / sqrt(Psi).
The constant C was inferred from two points; test it by pushing the TFRS chart
(Psi ~ 10, so the rule predicts failure near occupancy 6) up the axis and
finding where ARI actually crosses the 0.10 budget.
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

def psi_of(w, occ=2.0):
    da,ca,_,_,_,_,_,_ = w.docs_for_occupancy(occ)
    D=len(da); pc=Counter(zip(da.tolist(),ca.tolist()))
    post=Counter(np.concatenate([da,ca]).tolist()); tot=sum(post.values())
    f={a:c/tot for a,c in post.items()}
    return sum((n/D)**2/(f[a]*f[b]) for (a,b),n in pc.items()
               if f.get(a,0)>0 and f.get(b,0)>0)

SEEDS=[11,22,33]
print("="*84, flush=True)
print("CALIBRATION: where does ARI cross 0.10 on the structured chart?", flush=True)
print("="*84, flush=True)
psi = float(np.mean([psi_of(TFRSWorld(s,m=300)) for s in SEEDS]))
print("  TFRS chart: Psi = {:.1f}   rule with C=20 predicts occ <= {:.1f}".format(
      psi, 20/np.sqrt(psi)), flush=True)
print(flush=True)
print("  {:>5} {:>14} {:>22} {:>8}".format("occ","rep.edges","ARI (95% CI)","upper"), flush=True)
cross=None; prev=None
for occ in (2,4,6,8,12,16,24):
    A=[];R=[]
    for sd in SEEDS:
        w=TFRSWorld(sd,m=300)
        da,ca,d,c,_,_,_,_=w.docs_for_occupancy(occ)
        ap=Counter(zip(d.tolist(),c.tolist()))
        R.append(sum(1 for v in ap.values() if v>=2))
        A.append(attack(w,d,c))
    m,h,_=ci95(A); up=m+h
    flag="" if up<0.10 else "  <== OVER"
    print("  {:>5} {:>14.0f} {:>12.4f} +/- {:.4f} {:>8.4f}{}".format(
          occ,np.mean(R),m,h,up,flag), flush=True)
    if cross is None and prev is not None and up>=0.10:
        lo_o,lo_u=prev
        cross = lo_o + (0.10-lo_u)*(occ-lo_o)/(up-lo_u)
    prev=(occ,up)
print(flush=True)
if cross:
    print("  ARI crosses 0.10 at occupancy ~ {:.1f}".format(cross), flush=True)
    print("  implied constant C = occ*sqrt(Psi) = {:.1f}".format(cross*np.sqrt(psi)), flush=True)
else:
    print("  did not cross 0.10 in the tested range", flush=True)
