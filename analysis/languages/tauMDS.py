import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from common.utils import *
from common.ranking import kendall_tau, compute_distance_matrix
from analysis.languages.anova import build_long_df
from analysis.languages.used import normalize_languages


SHOW_MULTI_LANG = False  # include tasks with >1 normalized language


def build_task_langs(long):
    rows = long[["task_name", "languages"]].drop_duplicates(subset=["task_name"])
    return {
        r.task_name: tuple(sorted(normalize_languages(r.languages)))
        for r in rows.itertuples()
    }


def plot_mds(pivot, task_langs, fn, show_multi=False):
    tasks_all = list(pivot.index)
    if show_multi:
        keep_tasks = tasks_all
    else:
        keep_tasks = [t for t in tasks_all if len(task_langs[t]) == 1]

    if len(keep_tasks) < 2:
        print(f"  not enough tasks to plot ({len(keep_tasks)}) — aborting")
        return

    pivot_kept = pivot.loc[keep_tasks]
    D = compute_distance_matrix(pivot_kept)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        n_init=8,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(D)

    # Color palette per language (single-lang tasks only colored; multi gets grey).
    single_langs = sorted(
        {task_langs[t][0] for t in keep_tasks if len(task_langs[t]) == 1}
    )
    if single_langs:
        cmap = plt.cm.tab20 if len(single_langs) <= 20 else plt.cm.gist_ncar
        palette = cmap(np.linspace(0, 1, max(len(single_langs), 1)))
        lang_to_color = {l: palette[i] for i, l in enumerate(single_langs)}
    else:
        lang_to_color = {}
    multi_color = (0.35, 0.35, 0.35, 1.0)

    fig, ax = plt.subplots(figsize=(15, 11))
    for (x, y), t in zip(coords, keep_tasks):
        langs = task_langs[t]
        if len(langs) == 1:
            color = lang_to_color[langs[0]]
            marker = "o"
            label_lang = langs[0]
        else:
            color = multi_color
            marker = "^"
            label_lang = "+".join(langs)
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=80,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.annotate(
            f"{t}|{label_lang}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6.5,
            zorder=4,
        )

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    title = "Uzdevumi MDS telpā pēc modeļu rangiem\n(distance = (1 − Kendall τ) / 2)"
    if not show_multi:
        title += "\nattēloti tikai 1-valodas uzdevumi"
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color="grey", lw=0.4, linestyle="--")
    ax.axvline(0, color="grey", lw=0.4, linestyle="--")
    ax.grid(True, alpha=0.2)

    # Legend: one entry per language (single-lang), plus a "multi" entry if shown.
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=lang_to_color[l],
            markeredgecolor="black",
            markersize=8,
            label=l,
        )
        for l in single_langs
    ]
    if show_multi:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=multi_color,
                markeredgecolor="black",
                markersize=8,
                label="vairākas",
            )
        )
    ncol = max(1, len(handles) // 14 + 1)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        ncol=ncol,
        title="Valoda",
    )

    plt.tight_layout()
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")


def main():
    df = get_multilingual_scores_dataframe(get_multilingual_models)
    long = build_long_df(df)
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    task_langs = build_task_langs(long)

    n_single = sum(1 for t in pivot.index if len(task_langs[t]) == 1)
    n_multi = sum(1 for t in pivot.index if len(task_langs[t]) > 1)
    print(f"Tasks: {len(pivot)} ({n_single} single-lang, {n_multi} multi-lang)")

    suffix = "" if not SHOW_MULTI_LANG else "_multi"
    plot_mds(
        pivot,
        task_langs,
        thesis_file(f"lang_task_kendall_tau_mds{suffix}.png"),
        show_multi=SHOW_MULTI_LANG,
    )


if __name__ == "__main__":
    main()
