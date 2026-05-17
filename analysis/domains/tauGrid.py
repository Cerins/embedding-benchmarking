import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from common.utils import *
from common.ranking import kendall_tau, permutation_pvals
from analysis.domains.anova import build_long_df


K_VALUES = [1, 3, 5, 10, 15, None]
K_LABELS = ["1", "3", "5", "10", "15", "Pilns"]
DEBUG_KS = {1}  # k values for which to print per-pair diagnostics


# Return domainxdomain mean tau, meaning task x task model rakning tau, where task are part of domain
def compute_domain_matrix(pivot, task_to_domain, k):
    tasks = list(pivot.index)
    domains = sorted({task_to_domain[t] for t in tasks if t in task_to_domain})
    domain_to_int = {d: i for i, d in enumerate(domains)}
    labeled_indices = [
        idx for idx, t in enumerate(tasks) if task_to_domain.get(t) is not None
    ]
    labeled_int = np.array(
        [domain_to_int[task_to_domain[tasks[idx]]] for idx in labeled_indices]
    )
    full_to_lbl = {fi: li for li, fi in enumerate(labeled_indices)}
    pair_pi, pair_pj, pair_pt = [], [], []
    pair_taus = defaultdict(list)
    all_taus = []
    debug = k in DEBUG_KS
    skipped = 0
    skip_reasons = defaultdict(int)
    examples_shown = 0

    if debug:
        print(f"\n=== DEBUG k={k} ===")

    for i in range(len(tasks)):
        t1 = tasks[i]
        d1 = task_to_domain.get(t1)
        if d1 is None:
            continue
        for j in range(i + 1, len(tasks)):
            t2 = tasks[j]
            d2 = task_to_domain.get(t2)
            if d2 is None:
                continue
            ctx = {} if debug else None
            tau = kendall_tau(pivot.loc[t1], pivot.loc[t2], k, debug_ctx=ctx)
            if debug and examples_shown < 8:
                print(f"  [{d1} vs {d2}] {t1} | {t2}")
                if np.isnan(tau):
                    print(
                        f"    -> SKIPPED: {ctx.get('reason')}, top_a={ctx.get('top_a')}, top_b={ctx.get('top_b')}"
                    )
                else:
                    print(f"    top_a={ctx.get('top_a')} top_b={ctx.get('top_b')}")
                    print(f"    union={ctx.get('union')}")
                    print(f"    capped_a={ctx.get('capped_a')}")
                    print(f"    capped_b={ctx.get('capped_b')}")
                    print(f"    tau={tau:.3f}")
                examples_shown += 1
            if np.isnan(tau):
                if debug:
                    skipped += 1
                    skip_reasons[ctx.get("reason", "unknown")] += 1
                continue
            all_taus.append(tau)
            key = tuple(sorted([d1, d2]))
            pair_taus[key].append(tau)
            pair_pi.append(full_to_lbl[i])
            pair_pj.append(full_to_lbl[j])
            pair_pt.append(tau)
    if debug:
        n_pairs_eval = len(all_taus) + skipped
        print(f"  total task pairs considered: {n_pairs_eval}")
        print(f"  tau computed: {len(all_taus)}   skipped: {skipped}")
        for reason, cnt in skip_reasons.items():
            print(f"    skip reason: {reason} ({cnt})")
        if all_taus:
            arr = np.array(all_taus)
            print(
                f"  tau distribution: min={arr.min():.3f}, mean={arr.mean():.3f}, "
                f"median={np.median(arr):.3f}, max={arr.max():.3f}"
            )
            uniq, counts = np.unique(np.round(arr, 4), return_counts=True)
            print(f"  unique tau values: {dict(zip(uniq.tolist(), counts.tolist()))}")
    matrix = pd.DataFrame(np.nan, index=domains, columns=domains)
    for (d1, d2), taus in pair_taus.items():
        avg = float(np.mean(taus))
        matrix.loc[d1, d2] = avg
        matrix.loc[d2, d1] = avg
    pval_matrix = pd.DataFrame(np.nan, index=domains, columns=domains)
    # Calculate permutation test if domain has better cohesion
    if pair_pi:
        pv = permutation_pvals(
            labeled_int,
            np.array(pair_pi),
            np.array(pair_pj),
            np.array(pair_pt, dtype=float),
            len(domains),
        )
        pval_matrix = pd.DataFrame(pv, index=domains, columns=domains)
        diag = np.diag(pval_matrix.values).copy()
        pval_matrix[:] = np.nan
        np.fill_diagonal(pval_matrix.values, diag)

    overall = float(np.mean(all_taus)) if all_taus else float("nan")
    return matrix, overall, pval_matrix


def plot_grid(pivot, task_to_domain, fn):
    panels = []
    for k in K_VALUES:
        m, avg, pvals = compute_domain_matrix(pivot, task_to_domain, k)
        panels.append((m, avg, pvals))

    finite_vals = [
        v for m, _, __ in panels for v in m.values.flatten() if not np.isnan(v)
    ]
    vmin = min(finite_vals) if finite_vals else -1.0
    vmax = max(finite_vals) if finite_vals else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    for ax, (m, avg, pvals), klabel in zip(axes, panels, K_LABELS):
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=9)
        for i in range(len(m.index)):
            for j in range(len(m.columns)):
                val = m.values[i, j]
                pval = pvals.values[i, j]
                if not np.isnan(val):
                    sig = (not np.isnan(pval)) and pval < 0.05
                    label = f"{val:.2f}"
                    if not np.isnan(pval):
                        label += f"\np={pval:.3f}"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                        fontweight="bold" if sig else "normal",
                    )
        ax.set_title(
            f"k = {klabel}   |   visu uzdevumu pāru vidējais τ = {avg:.3f}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, label="Kendall τ")

    plt.suptitle(
        "Vidējais Kendall τ starp uzdevumu modeļu rangiem pa domēniem",
        fontsize=12,
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")


def run(gm, suffix):
    df = get_scores_dataframe(gm)
    long = build_long_df(df)
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    task_to_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .set_index("task_name")["domain"]
        .to_dict()
    )
    plot_grid(pivot, task_to_domain, thesis_file(f"task_tau_domain_grid{suffix}.png"))


def main():
    run(get_models, "")


if __name__ == "__main__":
    main()
