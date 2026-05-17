import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from common.utils import *
from common.ranking import kendall_tau, permutation_pvals
from analysis.languages.anova import build_long_df
from analysis.languages.used import normalize_languages


K_VALUES = [1, 3, 5, 10, 15, None]
K_LABELS = ["1", "3", "5", "10", "15", "Pilns"]
DEBUG_KS = {1}


def build_task_langs(long):
    rows = long[["task_name", "languages"]].drop_duplicates(subset=["task_name"])
    return {
        r.task_name: tuple(sorted(normalize_languages(r.languages)))
        for r in rows.itertuples()
    }


def compute_lang_matrix(pivot, task_to_langs, k):
    tasks = [t for t in pivot.index if len(task_to_langs[t]) == 1]

    lang_task_count = {}
    for t in tasks:
        for l in task_to_langs[t]:
            lang_task_count[l] = lang_task_count.get(l, 0) + 1
    valid_langs = {l for l, c in lang_task_count.items() if c >= 2}
    tasks = [t for t in tasks if task_to_langs[t][0] in valid_langs]

    langs = sorted(valid_langs)
    lang_to_int = {l: i for i, l in enumerate(langs)}
    task_int_labels = np.array([lang_to_int[task_to_langs[t][0]] for t in tasks])
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
        langs_a = task_to_langs[t1]
        for j in range(i + 1, len(tasks)):
            t2 = tasks[j]
            langs_b = task_to_langs[t2]
            ctx = {} if debug else None
            tau = kendall_tau(pivot.loc[t1], pivot.loc[t2], k, debug_ctx=ctx)
            if debug and examples_shown < 8:
                la, lb = langs_a[0], langs_b[0]
                print(f"  [{la} vs {lb}] {t1} | {t2}")
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
            pair_pi.append(i)
            pair_pj.append(j)
            pair_pt.append(tau)
            for la in langs_a:
                for lb in langs_b:
                    key = tuple(sorted([la, lb]))
                    pair_taus[key].append(tau)

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

    matrix = pd.DataFrame(np.nan, index=langs, columns=langs)
    for (la, lb), taus in pair_taus.items():
        avg = float(np.mean(taus))
        matrix.loc[la, lb] = avg
        matrix.loc[lb, la] = avg

    pval_matrix = pd.DataFrame(np.nan, index=langs, columns=langs)
    if pair_pi:
        pv = permutation_pvals(
            task_int_labels,
            np.array(pair_pi),
            np.array(pair_pj),
            np.array(pair_pt, dtype=float),
            len(langs),
        )
        pval_matrix = pd.DataFrame(pv, index=langs, columns=langs)
        diag = np.diag(pval_matrix.values).copy()
        pval_matrix[:] = np.nan
        np.fill_diagonal(pval_matrix.values, diag)

    overall = float(np.mean(all_taus)) if all_taus else float("nan")
    return matrix, overall, pval_matrix


def plot_grid(pivot, task_to_langs, fn):
    panels = []
    for k in K_VALUES:
        m, avg, pvals = compute_lang_matrix(pivot, task_to_langs, k)
        panels.append((m, avg, pvals))

    finite_vals = [
        v for m, _, __ in panels for v in m.values.flatten() if not np.isnan(v)
    ]
    vmin = min(finite_vals) if finite_vals else -1.0
    vmax = max(finite_vals) if finite_vals else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    for ax, (m, avg, pvals), klabel in zip(axes, panels, K_LABELS):
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=8)
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
                        fontsize=5,
                        color="black",
                        fontweight="bold" if sig else "normal",
                    )
        ax.set_title(
            f"k = {klabel}   |   visu uzdevumu pāru vidējais τ = {avg:.3f}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, label="Kendall τ")

    # plt.suptitle(
    #     "Vidējais Kendall τ starp uzdevumu modeļu rangiem pa valodām"
    #     "  —  tikai vienas valodas uzdevumi (≥2 uzdevumi valodā)",
    #     fontsize=12, y=1.00,
    # )

    plt.tight_layout()
    plt.savefig(fn, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")


def main():
    df = get_multilingual_scores_dataframe(get_multilingual_models)
    long = build_long_df(df)
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    task_to_langs = build_task_langs(long)

    n_single = sum(1 for t in pivot.index if len(task_to_langs[t]) == 1)
    n_multi = sum(1 for t in pivot.index if len(task_to_langs[t]) > 1)
    print(f"Tasks: {len(pivot)} ({n_single} single-lang, {n_multi} multi-lang)")

    plot_grid(pivot, task_to_langs, thesis_file("lang_task_tau_lang_grid.png"))


if __name__ == "__main__":
    main()
