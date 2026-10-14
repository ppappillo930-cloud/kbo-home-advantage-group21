"""Small, dependency-light statistics helpers (numpy/scipy only)."""
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    tc = stats.t.ppf(0.5 + conf / 2, n - 1)
    return dict(n=n, mean=m, sd=s, se=se, lo=m - tc * se, hi=m + tc * se)


def one_sample(x, alt="greater"):
    x = np.asarray(x, float)
    r = mean_ci(x)
    t, p = stats.ttest_1samp(x, 0.0, alternative=alt)
    w = stats.wilcoxon(x, alternative=alt)
    sw = stats.shapiro(x)
    r.update(t=float(t), p=float(p), wilcoxon_p=float(w.pvalue), shapiro_p=float(sw.pvalue),
             cohen_d=r["mean"] / r["sd"], pct_positive=float((x > 0).mean()))
    return r


def welch(a, b, alt="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alt)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tc = stats.t.ppf(0.975, dfw)
    d = a.mean() - b.mean()
    F = va / vb
    pF = 2 * min(stats.f.cdf(F, na - 1, nb - 1), 1 - stats.f.cdf(F, na - 1, nb - 1))
    lev = stats.levene(a, b)
    mw = stats.mannwhitneyu(a, b, alternative=alt)
    return dict(n_a=na, n_b=nb, mean_a=a.mean(), mean_b=b.mean(), diff=d, lo=d - tc * se, hi=d + tc * se,
                t=float(t), df=float(dfw), p=float(p), F=float(F), F_p=float(pF), levene_p=float(lev.pvalue),
                mannwhitney_p=float(mw.pvalue))


def ols(y, X, names=None, add_const=True, robust=True, cluster=None):
    """OLS with classical / HC1 / cluster-robust standard errors. Returns dict with a coefficient table."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    sse = e @ e
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    if cluster is not None:
        g = pd.Series(cluster).astype(str).values
        G = len(np.unique(g))
        meat = np.zeros((k, k))
        for gi in np.unique(g):
            s = X[g == gi].T @ e[g == gi]
            meat += np.outer(s, s)
        V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
        dfr = G - 1
        se_type = f"cluster({G})"
    elif robust:
        meat = (X * e[:, None] ** 2).T @ X
        V = XtX_inv @ meat @ XtX_inv * n / (n - k)
        dfr = n - k
        se_type = "HC1"
    else:
        V = XtX_inv * sse / (n - k)
        dfr = n - k
        se_type = "classical"
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), dfr)
    tc = stats.t.ppf(0.975, dfr)
    tab = pd.DataFrame(dict(coef=b, se=se, t=t, p=p, lo=b - tc * se, hi=b + tc * se), index=names)
    # overall F (classical)
    if k > 1:
        F = ((sst - sse) / (k - 1)) / (sse / (n - k))
        Fp = stats.f.sf(F, k - 1, n - k)
    else:
        F, Fp = np.nan, np.nan
    return dict(table=tab, r2=r2, adj_r2=adj, n=n, k=k, F=F, F_p=Fp, se_type=se_type, resid=e, fitted=X @ b)


def corr(x, y):
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)
