#!/usr/bin/env python3
"""
A structured chart of accounts in place of the flat Zipf model.

The flat model drew both sides of every entry uniformly at random from all
accounts.  Real books do not look like that, and the differences all point the
same way -- towards more structure, which is what a spectral adversary eats:

  * a five-category hierarchy (asset / liability / equity / revenue / expense)
    with codes that share a prefix within a category
  * a handful of HUB accounts -- cash, bank, receivables, payables, VAT in and
    out -- that appear on one side of most entries
  * entries drawn from accounting ARCHETYPES (credit sale, cash purchase,
    receipt, payment, payroll, depreciation ...) rather than random pairs, so
    an expense is posted against cash or payables, never against another
    expense
  * normal balances respected: assets and expenses debited, liabilities,
    equity and revenue credited

Everything still varies with the seed.
"""
import numpy as np
from harness import alloc_hamilton

# category -> (share of the chart, leading digit)
CATS = [("asset", 0.25, 1), ("liab", 0.15, 2), ("equity", 0.05, 3),
        ("revenue", 0.15, 4), ("expense", 0.40, 5)]
N_CC, N_DT = 100, 10


class TFRSWorld:
    def __init__(self, seed, m=300, K=None, alpha=10_000,
                 hub_share=0.55, cat_skew=1.0, n_cc=5, n_dt=3,
                 pool_cc=N_CC, pool_dt=N_DT):
        self.rng = np.random.default_rng(seed)
        self.seed, self.m = seed, m
        self.K = (alpha - m) if K is None else K

        # ---- lay out the chart by category
        self.cat = np.empty(m, dtype=object)
        i = 0
        self.by_cat = {}
        for name, share, _digit in CATS:
            n = max(2, int(round(share * m)))
            n = min(n, m - i) if name == CATS[-1][0] else n
            idx = np.arange(i, min(i + n, m))
            self.by_cat[name] = idx
            for j in idx:
                self.cat[j] = name
            i += n
        if i < m:                      # remainder to expenses
            extra = np.arange(i, m)
            self.by_cat["expense"] = np.concatenate([self.by_cat["expense"],
                                                     extra])
            for j in extra:
                self.cat[j] = "expense"

        # ---- hubs: first account of the main categories
        a, l = self.by_cat["asset"], self.by_cat["liab"]
        self.HUB = dict(cash=int(a[0]), bank=int(a[1]), ar=int(a[2]),
                        vat_in=int(a[3]), ap=int(l[0]), vat_out=int(l[1]),
                        accr=int(l[2]))

        # ---- within-category popularity (a few accounts dominate)
        self.w_in_cat = {}
        for name, idx in self.by_cat.items():
            w = 1.0 / np.arange(1, len(idx) + 1) ** cat_skew
            self.w_in_cat[name] = w / w.sum()

        # ---- archetypes: (debit spec, credit spec, weight)
        #      spec is either ("hub", key) or ("cat", category)
        H, C = "hub", "cat"
        self.arch = [
            # high-volume operating cycle
            ((H, "ar"),   (C, "revenue"), 0.14),   # credit sale
            ((H, "bank"), (C, "revenue"), 0.10),   # cash sale
            ((C, "expense"), (H, "ap"),   0.15),   # purchase on credit
            ((C, "expense"), (H, "bank"), 0.12),   # purchase paid
            ((C, "expense"), (H, "cash"), 0.06),   # petty cash
            ((H, "bank"), (H, "ar"),      0.09),   # receipt from customer
            ((H, "ap"),   (H, "bank"),    0.08),   # payment to supplier
            ((C, "expense"), (H, "accr"), 0.04),   # accrual / payroll
            ((H, "vat_in"), (H, "ap"),    0.03),   # input VAT
            ((C, "revenue"), (H, "vat_out"), 0.03),  # output VAT
            # lower-volume entries that reach the rest of the chart -- without
            # these, non-current assets, long-term liabilities and equity are
            # never posted to and the chart has dead accounts, which distorts
            # any clustering measurement
            ((C, "asset"), (H, "bank"),   0.04),   # asset purchase
            ((C, "asset"), (H, "ap"),     0.03),   # asset on credit
            ((H, "bank"), (C, "liab"),    0.03),   # borrowing
            ((C, "liab"), (H, "bank"),    0.03),   # repayment
            ((H, "bank"), (C, "equity"),  0.02),   # capital injection
            ((C, "equity"), (H, "bank"),  0.015),  # dividend
            ((C, "expense"), (C, "asset"), 0.02),  # depreciation
            ((C, "asset"), (C, "revenue"), 0.01),  # accrued income
        ]
        w = np.array([x[2] for x in self.arch], dtype=float)
        self.arch_w = w / w.sum()
        self.hub_share = hub_share

        # ---- realised account frequency, from the archetype model
        cnt = np.zeros(m)
        for (ds, cs, _), pw in zip(self.arch, self.arch_w):
            for spec in (ds, cs):
                if spec[0] == H:
                    cnt[self.HUB[spec[1]]] += pw
                else:
                    idx = self.by_cat[spec[1]]
                    cnt[idx] += pw * self.w_in_cat[spec[1]]
        self.f = cnt / cnt.sum()

        self.k = alloc_hamilton(self.f, self.K)
        self.Kp = int(self.k.sum())
        self.start = np.concatenate([[0], np.cumsum(self.k)[:-1]])
        self.owner = np.repeat(np.arange(m), self.k)

        self.cc_of = [self.rng.choice(pool_cc, n_cc, replace=False)
                      for _ in range(m)]
        self.dt_of = [self.rng.choice(pool_dt, n_dt, replace=False)
                      for _ in range(m)]
        self.n_cc, self.n_dt = n_cc, n_dt

    def delta(self):
        return 0.5 * float(np.abs(self.f - self.k / self.Kp).sum())

    def _pick(self, spec, n):
        if spec[0] == "hub":
            return np.full(n, self.HUB[spec[1]], dtype=np.int64)
        idx = self.by_cat[spec[1]]
        return idx[self.rng.choice(len(idx), size=n,
                                   p=self.w_in_cat[spec[1]])]

    def docs(self, n_docs):
        r = self.rng
        t = r.choice(len(self.arch), size=n_docs, p=self.arch_w)
        da = np.empty(n_docs, dtype=np.int64)
        ca = np.empty(n_docs, dtype=np.int64)
        for ai in range(len(self.arch)):
            sel = np.nonzero(t == ai)[0]
            if len(sel) == 0:
                continue
            ds, cs, _ = self.arch[ai]
            da[sel] = self._pick(ds, len(sel))
            ca[sel] = self._pick(cs, len(sel))
        d_al = self.start[da] + r.integers(0, self.k[da])
        c_al = self.start[ca] + r.integers(0, self.k[ca])
        d_cc = np.array([self.cc_of[x][r.integers(self.n_cc)] for x in da])
        d_dt = np.array([self.dt_of[x][r.integers(self.n_dt)] for x in da])
        c_cc = np.array([self.cc_of[x][r.integers(self.n_cc)] for x in ca])
        c_dt = np.array([self.dt_of[x][r.integers(self.n_dt)] for x in ca])
        return da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt

    def docs_for_occupancy(self, occ):
        return self.docs(int(occ * self.Kp / 2))

    def describe(self):
        top = np.argsort(-self.f)[:6]
        return ("m={} K'={} hubs={} top-6 share={:.1%} max f={:.3%} "
                "min f={:.4%}".format(
                    self.m, self.Kp, len(self.HUB),
                    float(self.f[top].sum()), float(self.f.max()),
                    float(self.f.min())))
