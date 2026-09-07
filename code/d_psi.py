"""
Closed-form predictor test.

For an account pair (a,b) posted n_ab times, the alias-pair space is
k(a)*k(b), so repeated alias edges follow a birthday count:
    E[repeats_ab] ~ n_ab^2 / (2 k(a) k(b))
Summing, with k(a) = f(a)K and n_ab = D p_ab where D = occ*K/2:

    E[repeats] ~ (occ/2)^2 * Psi / 2,    Psi = sum_ab p_ab^2 / (f(a) f(b))

Psi is a property of the chart alone -- computable from a firm's own books.
Check the prediction against measured repeats.
"""
import numpy as np
from collections import Counter
from harness import World
from harness_tfrs import TFRSWorld

def psi_and_repeats(w, occ):
    da,ca,d_al,c_al,_,_,_,_ = w.docs_for_occupancy(occ)
    D = len(da)
    pc = Counter(zip(da.tolist(), ca.tolist()))
    post = Counter(np.concatenate([da,ca]).tolist())
    tot = sum(post.values())
    f = {a: c/tot for a,c in post.items()}
    psi = 0.0
    for (a,b),n in pc.items():
        p = n/D
        if f.get(a,0)>0 and f.get(b,0)>0:
            psi += p*p/(f[a]*f[b])
    pred = (D**2/ (w.Kp**2)) * psi / 2 * (w.Kp**2) / (w.Kp**2)   # keep explicit
    # E[repeats] = sum_ab n_ab^2 / (2 k(a) k(b)) computed directly too
    direct = 0.0
    for (a,b),n in pc.items():
        ka, kb = int(w.k[a]), int(w.k[b])
        if ka>0 and kb>0:
            direct += n*n/(2.0*ka*kb)
    ap = Counter(zip(d_al.tolist(), c_al.tolist()))
    meas = sum(1 for v in ap.values() if v>=2)
    return psi, direct, meas

SEEDS=[11,22,33]
print("="*92, flush=True)
print("CLOSED-FORM PREDICTOR for repeated alias edges", flush=True)
print("="*92, flush=True)
print("  {:<26} {:>5} {:>10} {:>12} {:>12} {:>8}".format(
      "world","occ","Psi","predicted","measured","ratio"), flush=True)
rows=[]
for label, mk in (
    ("TFRS cat_skew=1.0", lambda s: TFRSWorld(s,m=300,cat_skew=1.0)),
    ("TFRS cat_skew=2.6", lambda s: TFRSWorld(s,m=300,cat_skew=2.6)),
    ("flat 1600 tx types", lambda s: World(s,m=300,s=1.0,n_types=1600)),
    ("flat  400 tx types", lambda s: World(s,m=300,s=1.0,n_types=400)),
    ("flat  100 tx types", lambda s: World(s,m=300,s=1.0,n_types=100)),
):
    for occ in (1.0, 2.0, 4.0):
        P,Dd,M=[],[],[]
        for sd in SEEDS:
            p,d,m = psi_and_repeats(mk(sd), occ)
            P.append(p); Dd.append(d); M.append(m)
        pm,dm,mm = np.mean(P), np.mean(Dd), np.mean(M)
        rows.append((dm,mm))
        print("  {:<26} {:>5.1f} {:>10.1f} {:>12.1f} {:>12.1f} {:>8}".format(
              label,occ,pm,dm,mm,
              "{:.2f}".format(mm/dm) if dm>0.5 else "-"), flush=True)
a=np.array([r[0] for r in rows]); b=np.array([r[1] for r in rows])
ok=a>0.5
print(flush=True)
print("  Pearson corr(predicted, measured) = {:+.4f}  over {} points".format(
      float(np.corrcoef(a[ok],b[ok])[0,1]), int(ok.sum())), flush=True)
print("  mean ratio measured/predicted     = {:.2f}".format(
      float((b[ok]/a[ok]).mean())), flush=True)
