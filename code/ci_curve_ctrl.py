"""
The 0.88 figure came from a run using FULL KMeans, dims=64, refine=3.
The harness uses MiniBatchKMeans, dims=32, refine=1 -- validated equivalent at
occupancy 1, but NEVER at high occupancy.  Before concluding the curve
plateaus, check whether the gap is seed variance or attack strength.
"""
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from harness import World, ci95

def attack(w, u, v, dims, refine, fast):
    n = w.Kp
    C = sp.coo_matrix((np.ones(2*len(u)), (np.concatenate([u,v]),
        np.concatenate([v,u]))), shape=(n,n)).tocsr()
    C.data = np.log1p(C.data)
    seen = np.asarray((C != 0).sum(axis=1)).ravel() > 0
    idx = np.nonzero(seen)[0]; Cs = C[idx][:, idx]
    U,S,_ = svds(Cs.asfptype(), k=min(dims, len(idx)-1), random_state=w.seed)
    X = U*S; nn = np.linalg.norm(X,axis=1,keepdims=True); nn[nn==0]=1
    KM = (lambda k: MiniBatchKMeans(k, n_init=3, random_state=w.seed, batch_size=1024)) \
         if fast else (lambda k: KMeans(k, n_init=2, random_state=w.seed))
    lab = KM(w.m).fit_predict(X/nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)),(np.arange(len(idx)),lab)),
                          shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs@H).todense())
        nn = np.linalg.norm(Q,axis=1,keepdims=True); nn[nn==0]=1
        lab = KM(w.m).fit_predict(Q/nn)
    return float(adjusted_rand_score(w.owner[idx], lab))

SEEDS=[11,22,33]
OCC=103.0
print("CONTROL at occupancy {} (m=300, K=9700, 400 types)".format(OCC), flush=True)
print("  {:<38} {:>22}  per-seed".format("attack configuration","ARI (95% CI)"), flush=True)
for label, dims, refine, fast in (
        ("harness: MiniBatch, dims=32, refine=1", 32, 1, True),
        ("original: full KMeans, dims=64, refine=3", 64, 3, False)):
    vals=[]
    t0=time.time()
    for sd in SEEDS:
        w = World(sd, m=300, s=1.0, n_types=400)
        _,_,d,c,_,_,_,_ = w.docs_for_occupancy(OCC)
        vals.append(attack(w, d, c, dims, refine, fast))
    m,h,n = ci95(vals)
    print("  {:<38} {:>10.4f} +/- {:.4f}  {}   [{:.0f}s]".format(
        label, m, h, " ".join("{:.3f}".format(v) for v in vals), time.time()-t0), flush=True)
print("done", flush=True)
