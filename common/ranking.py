import numpy as np
from scipy.stats import kendalltau


# D[i, j] = (1 - tau(task_i, task_j)) / 2, in [0, 1].
def compute_distance_matrix(pivot):
    tasks = list(pivot.index)
    n = len(tasks)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tau = kendall_tau(pivot.iloc[i], pivot.iloc[j])
            if np.isnan(tau):
                raise Exception("Huh")
            else:
                d = (1.0 - tau) / 2.0
            D[i, j] = D[j, i] = d
    return D


# Label-shuffling permutation test (one-sided upper tail).
# p[a,b] = fraction of label permutations where the shuffled cell mean >= observed mean.
def permutation_pvals(
    task_int_labels, pair_i, pair_j, pair_taus_arr, n_labels, n_perm=1_000, seed=42
):
    D = n_labels

    def _cell_means(labels):
        li = labels[pair_i]
        lj = labels[pair_j]
        min_l = np.minimum(li, lj)
        max_l = np.maximum(li, lj)
        cell_key = min_l * D + max_l
        tau_sum = np.bincount(cell_key, weights=pair_taus_arr, minlength=D * D)
        cnt = np.bincount(cell_key, minlength=D * D)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tau_sum / cnt, np.nan).reshape(D, D)

    observed = _cell_means(task_int_labels)
    valid = ~np.isnan(observed)
    exceedances = np.zeros((D, D))

    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        perm_means = _cell_means(rng.permutation(task_int_labels))
        mask = valid & ~np.isnan(perm_means)
        exceedances[mask] += perm_means[mask] >= observed[mask]

    pvals = np.full((D, D), np.nan)
    pvals[valid] = exceedances[valid] / n_perm
    return pvals


# Kendall tau between two tasks' model rankings.
# k=None: standard tau on all models present in both tasks.
# finite k: union of top-k models from each task, with ranks of any model
# outside its task's top-k capped at k+1 (tied).
def kendall_tau(scores_a, scores_b, k=None, debug_ctx=None):
    common = scores_a.index.intersection(scores_b.index)
    a = scores_a.loc[common].dropna()
    b = scores_b.loc[common].dropna()
    common = a.index.intersection(b.index)
    if len(common) < 2:
        if debug_ctx is not None:
            debug_ctx["reason"] = "too few common models"
        return np.nan
    a = a.loc[common]
    b = b.loc[common]
    if k is None or k >= len(a):
        tau, _ = kendalltau(a, b)
        if debug_ctx is not None:
            debug_ctx["mode"] = "full"
            debug_ctx["tau"] = tau
        return tau
    rank_a = a.rank(ascending=False, method="min")
    rank_b = b.rank(ascending=False, method="min")
    top_a = rank_a[rank_a <= k].index.tolist()
    top_b = rank_b[rank_b <= k].index.tolist()
    union = rank_a.index[(rank_a <= k) | (rank_b <= k)]
    if len(union) < 2:
        if debug_ctx is not None:
            debug_ctx["mode"] = f"top-{k} (union<2 -> +1)"
            debug_ctx["top_a"] = top_a
            debug_ctx["top_b"] = top_b
            debug_ctx["tau"] = 1.0
        return 1.0
    capped_a = rank_a.loc[union].clip(upper=k + 1)
    capped_b = rank_b.loc[union].clip(upper=k + 1)
    if capped_a.nunique() < 2 or capped_b.nunique() < 2:
        if capped_a.equals(capped_b):
            if debug_ctx is not None:
                debug_ctx["mode"] = f"top-{k} (all-tied identical -> +1)"
                debug_ctx["tau"] = 1.0
            return 1.0
        if debug_ctx is not None:
            debug_ctx["reason"] = "capped ranks all tied on one side"
        return np.nan
    tau, _ = kendalltau(capped_a, capped_b)
    if debug_ctx is not None:
        debug_ctx["mode"] = f"top-{k}"
        debug_ctx["top_a"] = top_a
        debug_ctx["top_b"] = top_b
        debug_ctx["union"] = list(union)
        debug_ctx["capped_a"] = capped_a.to_dict()
        debug_ctx["capped_b"] = capped_b.to_dict()
        debug_ctx["tau"] = tau
    return tau
