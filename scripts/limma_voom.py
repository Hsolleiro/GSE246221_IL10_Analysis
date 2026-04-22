"""
Python implementation of limma-voom for bulk RNA-seq differential expression.

Implements:
- TMM normalization (Robinson & Oshlack 2010)
- voom precision weights (Law et al. 2014)
- Empirical Bayes moderated t-statistics (Smyth 2004)
- Multiple contrasts with Benjamini-Hochberg FDR

Reference implementations cross-checked against edgeR/limma R packages.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import polygamma
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# TMM normalization (Robinson & Oshlack 2010)
# =============================================================================

def tmm_norm_factors(counts, log_ratio_trim=0.30, sum_trim=0.05):
    """Calculate TMM normalization factors (edgeR::calcNormFactors)."""
    counts = np.asarray(counts, dtype=float)
    lib = counts.sum(axis=0)
    n = counts.shape[1]
    cpm_like = counts / lib[np.newaxis, :]
    uq = np.quantile(cpm_like, 0.75, axis=0)
    ref_col = int(np.argmin(np.abs(uq - uq.mean())))
    ref = counts[:, ref_col]
    ref_lib = lib[ref_col]
    nf = np.ones(n)
    for i in range(n):
        if i == ref_col:
            continue
        obs = counts[:, i]
        obs_lib = lib[i]
        nz = (obs > 0) & (ref > 0)
        if nz.sum() == 0:
            continue
        o = obs[nz]; r = ref[nz]
        log_r = np.log2((o / obs_lib) / (r / ref_lib))
        abs_e = 0.5 * (np.log2(o / obs_lib) + np.log2(r / ref_lib))
        v = (obs_lib - o) / (obs_lib * o) + (ref_lib - r) / (ref_lib * r)
        finite = np.isfinite(log_r) & np.isfinite(abs_e)
        log_r, abs_e, v = log_r[finite], abs_e[finite], v[finite]
        k = len(log_r)
        if k == 0:
            continue
        low_m = int(np.floor(k * log_ratio_trim)) + 1
        hi_m  = k + 1 - low_m
        low_a = int(np.floor(k * sum_trim)) + 1
        hi_a  = k + 1 - low_a
        rm = pd.Series(log_r).rank(method='average').values
        ra = pd.Series(abs_e).rank(method='average').values
        keep = (rm >= low_m) & (rm <= hi_m) & (ra >= low_a) & (ra <= hi_a)
        if keep.sum() == 0:
            continue
        w = 1.0 / v[keep]
        nf[i] = 2.0 ** (np.sum(w * log_r[keep]) / np.sum(w))
    return nf / np.exp(np.mean(np.log(nf)))


# =============================================================================
# voom (Law et al. 2014)
# =============================================================================

def voom(counts, design, lib_size=None, norm_factors=None, span=0.5, plot_path=None):
    """
    Transform counts to log2-CPM with precision weights.

    counts: (n_genes, n_samples) integer matrix
    design: (n_samples, n_coefs) design matrix
    lib_size: library sizes (default: column sums of counts)
    norm_factors: TMM factors (default: ones)
    span: LOWESS span for mean-variance trend
    plot_path: if provided, save voom plot (sqrt(residual sd) vs log-count mean)

    Returns:
      y: log2-CPM matrix (n_genes x n_samples)
      weights: precision weights (same shape)
    """
    counts = np.asarray(counts, dtype=float)
    n_genes, n_samples = counts.shape
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    if norm_factors is None:
        norm_factors = np.ones(n_samples)
    eff_lib = lib_size * norm_factors

    # log2-CPM with voom's prior count: 0.5 added to counts, 1 added to lib (in counts scale)
    y = np.log2((counts + 0.5) / (eff_lib + 1.0) * 1e6)

    # Initial lmFit (OLS per gene)
    # beta_hat = (X'X)^-1 X' y^T, residuals via hat matrix
    XtX_inv = np.linalg.inv(design.T @ design)
    beta = (XtX_inv @ design.T @ y.T).T  # (n_genes, n_coefs)
    fitted = (design @ beta.T).T          # (n_genes, n_samples)
    resid = y - fitted
    df_resid = n_samples - np.linalg.matrix_rank(design)
    # sigma per gene
    sigma2 = (resid ** 2).sum(axis=1) / df_resid
    sigma = np.sqrt(sigma2)
    # Amean: average log-CPM per gene (standard voom x-axis)
    Amean = y.mean(axis=1)

    # log-count mean (x-axis of mean-variance plot):
    # fitted values in log-counts (NOT log-CPM) — voom uses log2(fitted_counts + 0.5)
    # Specifically: convert fitted back to log2(count) scale assuming the same lib size
    # Law et al. use: mean_log_count = log2(fitted_cpm) + log2(lib_size * nf / 1e6)
    # Simplified: x = fitted_log_cpm_mean + log2(mean_eff_lib / 1e6)
    # edgeR's approach: x_i = mean(log_count_per_gene) approximated as:
    log_count_mean = Amean + np.log2(eff_lib.mean() / 1e6 + 1)

    # Fit LOWESS to sqrt(sigma) vs log_count_mean
    # Filter out genes with sigma = 0 or non-finite
    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(log_count_mean)
    sort_idx = np.argsort(log_count_mean[ok])
    x_sorted = log_count_mean[ok][sort_idx]
    y_sorted = np.sqrt(sigma[ok][sort_idx])
    lo = lowess(y_sorted, x_sorted, frac=span, return_sorted=True)
    # Interpolate back to all genes (including the ones filtered out for fitting)
    from scipy.interpolate import interp1d
    interp = interp1d(lo[:, 0], lo[:, 1], bounds_error=False,
                      fill_value=(lo[0, 1], lo[-1, 1]))

    # Predict sigma^2 per gene-per-sample based on fitted values
    # Per-observation log-count: log2(fitted_cpm_{ij}) + log2(eff_lib_j / 1e6)
    fitted_log_count = fitted + np.log2(eff_lib[np.newaxis, :] / 1e6 + 1)
    # Predicted sqrt(sigma) for each (gene, sample)
    pred_sqrt_sigma = interp(fitted_log_count)
    # Precision weight = 1 / sigma^2 = 1 / pred_sqrt_sigma^4
    weights = 1.0 / (pred_sqrt_sigma ** 4)

    if plot_path is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(log_count_mean, np.sqrt(sigma), s=3, alpha=0.3, rasterized=True)
        xx = np.linspace(log_count_mean.min(), log_count_mean.max(), 200)
        ax.plot(xx, interp(xx), 'r-', lw=2, label='LOWESS fit')
        ax.set_xlabel('log2(count mean + 0.5)')
        ax.set_ylabel('sqrt(residual std dev)')
        ax.set_title('voom: mean-variance trend')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=140)
        plt.close()

    return y, weights, Amean, sigma, df_resid


# =============================================================================
# Weighted lmFit + empirical Bayes (Smyth 2004)
# =============================================================================

def lmfit_weighted(y, design, weights):
    """
    Fit a weighted linear model per gene.

    Returns:
      coefficients: (n_genes, n_coefs)
      stdev_unscaled: (n_genes, n_coefs)  — sqrt(diag((X'WX)^-1))
      sigma: (n_genes,) — residual standard deviation (weighted)
      df_residual: (n_genes,)
    """
    n_genes, n_samples = y.shape
    n_coefs = design.shape[1]
    df_resid = n_samples - np.linalg.matrix_rank(design)

    coef = np.zeros((n_genes, n_coefs))
    stdev_unsc = np.zeros((n_genes, n_coefs))
    sigma = np.zeros(n_genes)

    # Per-gene weighted least squares
    for g in range(n_genes):
        w = weights[g]
        sw = np.sqrt(w)
        Xw = design * sw[:, np.newaxis]
        yw = y[g] * sw
        try:
            XtX_inv = np.linalg.inv(Xw.T @ Xw)
        except np.linalg.LinAlgError:
            coef[g] = np.nan
            stdev_unsc[g] = np.nan
            sigma[g] = np.nan
            continue
        beta = XtX_inv @ Xw.T @ yw
        resid = yw - Xw @ beta
        s2 = (resid ** 2).sum() / df_resid
        coef[g] = beta
        stdev_unsc[g] = np.sqrt(np.diag(XtX_inv))
        sigma[g] = np.sqrt(s2)

    df_res_arr = np.full(n_genes, df_resid, dtype=float)
    return coef, stdev_unsc, sigma, df_res_arr


def _trigamma_inverse(y_val):
    """Invert trigamma function: find x such that polygamma(1, x) = y_val."""
    if y_val <= 0:
        return np.inf
    x = 1.0 / np.sqrt(y_val) if y_val > 1e-6 else 1e6
    x = max(x, 1e-6)
    for _ in range(200):
        val = polygamma(1, x)
        if not np.isfinite(val):
            break
        deriv = polygamma(2, x)
        if deriv == 0 or not np.isfinite(deriv):
            break
        step = (val - y_val) / deriv
        x_new = x - step
        if x_new <= 0:
            x_new = x / 2
        if abs(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
    return x


def fit_prior_variance(sigma, df_resid, covariate=None):
    """
    Estimate d0 and s0^2 for empirical Bayes shrinkage (Smyth 2004).
    Uses method of moments on the trigamma scale.
    Returns constant prior (no trend).
    """
    from scipy.special import digamma
    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(df_resid) & (df_resid > 0)
    z = np.log(sigma[ok] ** 2)
    df = df_resid[ok]

    e = z - digamma(df / 2) + np.log(df / 2)
    mean_e = e.mean()
    var_e = e.var(ddof=1) - polygamma(1, df / 2).mean()

    if var_e <= 0:
        return np.inf, np.exp(mean_e)

    d0_half = _trigamma_inverse(var_e)
    df0 = 2 * d0_half
    s02 = np.exp(mean_e - digamma(d0_half) + np.log(d0_half))
    return df0, s02


def fit_prior_variance_trended(sigma, df_resid, Amean, span=0.5):
    """
    Trended empirical Bayes: estimate s0^2 as a smooth function of Amean,
    while keeping df0 a single value (limma's trend=TRUE approach).

    This is more appropriate for RNA-seq data where low-expression genes
    have inherently higher variance than high-expression genes.

    Returns:
      df0: scalar
      s02_trend: array of length n_genes (one prior variance per gene, based on Amean)
    """
    from scipy.special import digamma
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d

    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(df_resid) & (df_resid > 0)
    z = np.log(sigma[ok] ** 2)
    df = df_resid[ok]
    A = Amean[ok]

    # Constant e: e_i = z_i - digamma(df_i/2) + log(df_i/2)
    # This removes the bias due to sampling, so e_i reflects log(s0^2_true) + noise
    e = z - digamma(df / 2) + np.log(df / 2)

    # Trend: fit LOWESS of e vs Amean
    sort_idx = np.argsort(A)
    A_sorted = A[sort_idx]
    e_sorted = e[sort_idx]
    lo = lowess(e_sorted, A_sorted, frac=span, return_sorted=True, it=3)
    interp = interp1d(lo[:, 0], lo[:, 1], bounds_error=False,
                      fill_value=(lo[0, 1], lo[-1, 1]))
    e_trend = interp(Amean)

    # Residuals from trend give us the prior variance df0 estimate
    e_resid = e - e_trend[ok]
    var_e = e_resid.var(ddof=1) - polygamma(1, df / 2).mean()
    if var_e <= 0:
        df0 = np.inf
    else:
        d0_half = _trigamma_inverse(var_e)
        df0 = 2 * d0_half

    # Prior variance per gene based on trend
    if np.isinf(df0):
        s02_trend = np.exp(e_trend)
    else:
        d0_half = df0 / 2
        s02_trend = np.exp(e_trend - digamma(d0_half) + np.log(d0_half))

    return df0, s02_trend


def fit_prior_variance_robust(sigma, df_resid, Amean=None, span=0.5, trend=True):
    """
    Robust empirical Bayes using Winsorization of outlier sigma values
    (approximation of Phipson et al. 2016's limma::squeezeVar(robust=TRUE)).

    Down-weights extreme-variance genes when estimating prior.
    """
    from scipy.special import digamma
    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(df_resid) & (df_resid > 0)
    z = np.log(sigma[ok] ** 2)
    df = df_resid[ok]
    e = z - digamma(df / 2) + np.log(df / 2)

    # Winsorize at 2nd and 98th percentile to get robust variance estimate
    lo_q, hi_q = np.percentile(e, [2, 98])
    e_wins = np.clip(e, lo_q, hi_q)
    var_e = e_wins.var(ddof=1) - polygamma(1, df / 2).mean()

    if trend and Amean is not None:
        # Trended with robust variance
        from statsmodels.nonparametric.smoothers_lowess import lowess
        from scipy.interpolate import interp1d
        A = Amean[ok]
        sort_idx = np.argsort(A)
        A_sorted = A[sort_idx]
        e_sorted = e_wins[sort_idx]  # use winsorized for fitting
        lo = lowess(e_sorted, A_sorted, frac=span, return_sorted=True, it=3)
        interp = interp1d(lo[:, 0], lo[:, 1], bounds_error=False,
                          fill_value=(lo[0, 1], lo[-1, 1]))
        e_trend = interp(Amean)
    else:
        e_trend_val = np.median(e_wins)
        e_trend = np.full(len(Amean), e_trend_val) if Amean is not None else e_trend_val

    if var_e <= 0:
        df0 = np.inf
    else:
        d0_half = _trigamma_inverse(var_e)
        df0 = 2 * d0_half

    if np.isinf(df0):
        s02_trend = np.exp(e_trend)
    else:
        d0_half = df0 / 2
        s02_trend = np.exp(e_trend - digamma(d0_half) + np.log(d0_half))

    return df0, s02_trend


def ebayes(coef, stdev_unsc, sigma, df_resid):
    """Apply empirical Bayes variance shrinkage and compute moderated t and p values."""
    df0, s02 = fit_prior_variance(sigma, df_resid)
    # Posterior s^2 (Smyth 2004 eq. 8)
    if np.isinf(df0):
        s2_post = np.full_like(sigma, s02)
        df_total = df_resid
    else:
        s2_post = (df0 * s02 + df_resid * sigma ** 2) / (df0 + df_resid)
        df_total = df_resid + df0
    sigma_post = np.sqrt(s2_post)
    # Moderated t per coefficient
    t = coef / (stdev_unsc * sigma_post[:, np.newaxis])
    # p-values (two-sided)
    p = 2 * stats.t.sf(np.abs(t), df_total[:, np.newaxis])
    return {
        'coefficients': coef,
        'stdev_unscaled': stdev_unsc,
        'sigma': sigma,
        'sigma_post': sigma_post,
        's2_prior': s02,
        'df_prior': df0,
        'df_total': df_total,
        't': t,
        'p_value': p,
    }


def contrasts_fit(fit_ebayes, contrasts_matrix):
    """
    Apply a contrasts matrix to an eBayes fit.
    contrasts_matrix: (n_coefs, n_contrasts)
    """
    coef = fit_ebayes['coefficients'] @ contrasts_matrix  # (n_genes, n_contrasts)
    # stdev_unscaled for each contrast: sqrt(c' (X'WX)^-1 c) needs refit; approximation:
    # For voom, use the full Sigma per gene — here we approximate as:
    # var(c'beta) = c' Cov(beta) c = c' (sigma^2 * (X'WX)^-1) c
    # We stored stdev_unsc = sqrt(diag((X'WX)^-1)), which doesn't support arbitrary contrasts.
    # For correctness, we re-solve. But in limma, contrasts.fit stores the full correlation.
    # Simplified for our case: if contrasts are single coefficients (identity contrast), unchanged.
    # For pairwise contrasts, var(b1 - b2) = var(b1) + var(b2) - 2*cov(b1,b2).
    # We need cov_unsc — recompute from (X'WX)^-1. We'll store that in lmfit_weighted next time.
    raise NotImplementedError("Use ebayes() with pre-constructed contrast design instead.")


# =============================================================================
# High-level function: fit model for a set of contrasts
# =============================================================================

def fit_contrasts_directly(y, weights, design, contrast_vectors, gene_names,
                            trend=True, robust=True, span=0.5):
    """
    For each contrast vector c, compute moderated t-statistic and p-value.

    trend: if True, use trended prior variance (recommended for RNA-seq).
    robust: if True, Winsorize for robust variance estimation.

    For a design matrix X = [G1, G2, ..., Gk] (group indicators, e.g., from ~0 + group),
    a contrast c (e.g., [1,-1,0,0,0,0] for G1-G2) is tested by:
        t = (c' beta) / sqrt(sigma_post^2 * c' (X'WX)^-1 c)
    """
    n_genes, n_samples = y.shape
    n_coefs = design.shape[1]
    df_resid = n_samples - np.linalg.matrix_rank(design)

    # Fit per-gene: store coef and XtWX_inv
    coef = np.zeros((n_genes, n_coefs))
    sigma = np.zeros(n_genes)
    cov_unsc = np.zeros((n_genes, n_coefs, n_coefs))

    for g in range(n_genes):
        w = weights[g]
        sw = np.sqrt(w)
        Xw = design * sw[:, np.newaxis]
        yw = y[g] * sw
        try:
            XtWX_inv = np.linalg.inv(Xw.T @ Xw)
        except np.linalg.LinAlgError:
            coef[g] = np.nan; sigma[g] = np.nan
            cov_unsc[g] = np.nan
            continue
        beta = XtWX_inv @ Xw.T @ yw
        resid = yw - Xw @ beta
        s2 = (resid ** 2).sum() / df_resid
        coef[g] = beta
        sigma[g] = np.sqrt(s2)
        cov_unsc[g] = XtWX_inv

    # Average expression per gene
    Amean = y.mean(axis=1)

    # Fit prior variance (trended + robust is the recommended default)
    df_resid_arr = np.full(n_genes, df_resid, dtype=float)
    if robust:
        df0, s02_arr = fit_prior_variance_robust(sigma, df_resid_arr, Amean=Amean,
                                                  span=span, trend=trend)
    elif trend:
        df0, s02_arr = fit_prior_variance_trended(sigma, df_resid_arr, Amean, span=span)
    else:
        df0, s02_const = fit_prior_variance(sigma, df_resid_arr)
        s02_arr = np.full(n_genes, s02_const)

    if np.isinf(df0):
        s2_post = s02_arr
        df_total = df_resid_arr
    else:
        s2_post = (df0 * s02_arr + df_resid * sigma ** 2) / (df0 + df_resid)
        df_total = df_resid_arr + df0
    sigma_post = np.sqrt(s2_post)

    # Apply each contrast
    results = {}
    from statsmodels.stats.multitest import multipletests
    for name, c in contrast_vectors.items():
        c = np.asarray(c, dtype=float)
        lfc = coef @ c
        var_unsc = np.array([c @ cov_unsc[g] @ c for g in range(n_genes)])
        se_unsc = np.sqrt(var_unsc)
        t = lfc / (se_unsc * sigma_post)
        p = 2 * stats.t.sf(np.abs(t), df_total)
        ok = np.isfinite(p)
        fdr = np.full(n_genes, np.nan)
        if ok.sum() > 0:
            _, fdr_ok, _, _ = multipletests(p[ok], method='fdr_bh')
            fdr[ok] = fdr_ok
        df_out = pd.DataFrame({
            'gene': gene_names,
            'logFC': lfc,
            'AveExpr': Amean,
            't': t,
            'P.Value': p,
            'adj.P.Val': fdr,
        })
        results[name] = df_out.sort_values('P.Value').reset_index(drop=True)

    return results, {'df_prior': df0, 's2_prior': s02_arr.mean() if hasattr(s02_arr,'mean') else s02_arr,
                     'sigma_post': sigma_post}


# =============================================================================
# F-test for longitudinal ANOVA-like contrast
# =============================================================================

def f_test_contrasts(y, weights, design, contrast_matrix, gene_names,
                      trend=True, robust=True, span=0.5):
    """
    F-test for a set of contrasts jointly (like limma's topTableF).
    contrast_matrix: (n_coefs, n_joint_contrasts) — tests H0: all contrasts = 0.

    Useful for ANOVA-like: "any difference across stages".
    """
    n_genes, n_samples = y.shape
    n_coefs = design.shape[1]
    df_resid = n_samples - np.linalg.matrix_rank(design)
    df_num = np.linalg.matrix_rank(contrast_matrix)

    coef = np.zeros((n_genes, n_coefs))
    sigma = np.zeros(n_genes)
    cov_unsc = np.zeros((n_genes, n_coefs, n_coefs))

    for g in range(n_genes):
        w = weights[g]
        sw = np.sqrt(w)
        Xw = design * sw[:, np.newaxis]
        yw = y[g] * sw
        try:
            XtWX_inv = np.linalg.inv(Xw.T @ Xw)
        except np.linalg.LinAlgError:
            coef[g] = np.nan; sigma[g] = np.nan
            cov_unsc[g] = np.nan
            continue
        beta = XtWX_inv @ Xw.T @ yw
        resid = yw - Xw @ beta
        s2 = (resid ** 2).sum() / df_resid
        coef[g] = beta
        sigma[g] = np.sqrt(s2)
        cov_unsc[g] = XtWX_inv

    Amean = y.mean(axis=1)
    df_resid_arr = np.full(n_genes, df_resid, dtype=float)
    if robust:
        df0, s02_arr = fit_prior_variance_robust(sigma, df_resid_arr, Amean=Amean,
                                                  span=span, trend=trend)
    elif trend:
        df0, s02_arr = fit_prior_variance_trended(sigma, df_resid_arr, Amean, span=span)
    else:
        df0, s02_const = fit_prior_variance(sigma, df_resid_arr)
        s02_arr = np.full(n_genes, s02_const)

    if np.isinf(df0):
        s2_post = s02_arr
        df_total = df_resid_arr
    else:
        s2_post = (df0 * s02_arr + df_resid * sigma ** 2) / (df0 + df_resid)
        df_total = df_resid_arr + df0

    C = contrast_matrix
    F_stat = np.zeros(n_genes)
    for g in range(n_genes):
        if not np.all(np.isfinite(cov_unsc[g])):
            F_stat[g] = np.nan
            continue
        Cb = C.T @ coef[g]
        CcC = C.T @ cov_unsc[g] @ C
        try:
            CcC_inv = np.linalg.inv(CcC)
        except np.linalg.LinAlgError:
            F_stat[g] = np.nan
            continue
        numer = Cb @ CcC_inv @ Cb / df_num
        F_stat[g] = numer / s2_post[g]

    p = stats.f.sf(F_stat, df_num, df_total)
    from statsmodels.stats.multitest import multipletests
    ok = np.isfinite(p)
    fdr = np.full(n_genes, np.nan)
    if ok.sum() > 0:
        _, fdr_ok, _, _ = multipletests(p[ok], method='fdr_bh')
        fdr[ok] = fdr_ok

    return pd.DataFrame({
        'gene': gene_names,
        'F': F_stat,
        'P.Value': p,
        'adj.P.Val': fdr,
        'AveExpr': y.mean(axis=1),
    }).sort_values('P.Value').reset_index(drop=True)
