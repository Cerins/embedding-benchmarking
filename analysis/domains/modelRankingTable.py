import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

from common.utils import *
from analysis.domains.anova import build_long_df


# Find best model
def _find_best_overall_model(pivot):
    return pivot.idxmax(axis=1).value_counts().idxmax()


# Perofrmance color, white for best, soft rose-red at 80
def _perf_color(rel_pct_vs_task_best):
    if np.isnan(rel_pct_vs_task_best) or rel_pct_vs_task_best >= -5.0:
        return (1.0, 1.0, 1.0)
    t = min(1.0, (-rel_pct_vs_task_best - 5.0) / 75.0)
    g = 1.0 - t * 0.55
    b = 1.0 - t * 0.55
    return (1.0, g, b)


# Hypen wrapping code
def _wrap_hyphens(name, width=18):
    parts = name.split("-")
    lines = []
    current = parts[0]
    for part in parts[1:]:
        if len(current) + 1 + len(part) <= width:
            current += "-" + part
        else:
            lines.append(current + "-")
            current = part
    lines.append(current)
    return "\n".join(lines)



def plot_task_model_ranking_table(long, fn):
    # Order
    task_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .sort_values(["domain", "task_name"])
    )
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]
    # All model list
    all_models = long["model_short"].unique().tolist()
    # Pivot: rows = tasks, cols = all models (any order for now)
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    pivot = pivot.loc[ordered_tasks]
    best_per_task = pivot.max(axis=1)
    # Best overall model: wins most individual tasks
    best_overall = _find_best_overall_model(pivot)
    win_count = int((pivot.idxmax(axis=1) == best_overall).sum())
    best_overall_disp = model_short_name(best_overall)
    n_tasks = len(ordered_tasks)
    n_models = pivot.shape[1]
    # Per row ordering
    ranked_models = [
        pivot.loc[task].sort_values(na_position="first").index.tolist()
        for task in ordered_tasks
    ]
    # Build display matrices indexed by (task_row, rank_col)
    score_mat = np.full((n_tasks, n_models), np.nan)
    model_mat = [["" for _ in range(n_models)] for _ in range(n_tasks)]
    rel_task_mat = np.full((n_tasks, n_models), np.nan)
    rel_best_mat = np.full((n_tasks, n_models), np.nan)
    for i, task in enumerate(ordered_tasks):
        best_score = best_per_task.iloc[i]
        best_overall_score = (
            pivot.loc[task, best_overall] if best_overall in pivot.columns else np.nan
        )
        for j, model in enumerate(ranked_models[i]):
            score = pivot.loc[task, model]
            score_mat[i, j] = score
            model_mat[i][j] = model_short_name(model)
            if not (np.isnan(score) or np.isnan(best_score) or best_score == 0):
                rel_task_mat[i, j] = (score - best_score) / best_score * 100
            if not (
                np.isnan(score)
                or np.isnan(best_overall_score)
                or best_overall_score == 0
            ):
                rel_best_mat[i, j] = (
                    (score - best_overall_score) / best_overall_score * 100
                )

    # Figure
    cell_w = 1.6
    cell_h = 0.85
    label_left = 5.2
    label_bottom = 0.8
    colorbar_right = 1.3
    fig_w = label_left + n_models * cell_w + colorbar_right
    fig_h = label_bottom + n_tasks * cell_h + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    l = label_left / fig_w
    b = label_bottom / fig_h
    w = (n_models * cell_w) / fig_w
    h = (n_tasks * cell_h) / fig_h
    ax = fig.add_axes([l, b, w, h])
    ax.set_xlim(0, n_models)
    ax.set_ylim(0, n_tasks)
    ax.invert_yaxis()

    # Cells drawing
    for i in range(n_tasks):
        for j in range(n_models):
            score = score_mat[i, j]
            r_task = rel_task_mat[i, j]
            r_best = rel_best_mat[i, j]
            mname = model_mat[i][j]

            color = _perf_color(r_task)
            rect = Rectangle((j, i), 1, 1, fc=color, ec="#b8b8b8", lw=0.5, zorder=1)
            ax.add_patch(rect)

            if not np.isnan(score):
                cell_txt = (
                    f"{_wrap_hyphens(mname)}\n"
                    f"{score:.3f}\n"
                    f"uzd:{r_task:+.1f}%\n"
                    f"lab:{r_best:+.1f}%"
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    cell_txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    family="monospace",
                    zorder=2,
                )

    ax.set_xticks([j + 0.5 for j in range(n_models)])
    ax.set_xticklabels(
        [f"vieta {j + 1}" for j in range(n_models)],
        fontsize=8,
    )
    ax.set_xlabel(
        "← sliktākais uzdevumā          labākais uzdevumā →", fontsize=8, labelpad=4
    )
    ax.set_yticks([i + 0.5 for i in range(n_tasks)])
    ax.set_yticklabels(task_labels, fontsize=7.5)
    ax.set_title(
        f"Uzdevums × Modelis  —  kolonnas = vieta uzdevumā (sliktākais→labākais)  |  rindas = uzdevumi pēc domēna\n"
        f"Kopumā labākais modelis: {best_overall_disp}  (uzvar {win_count}/{n_tasks} uzdevumos)\n"
        f"Šūna: modeļa nosaukums, nDCG@10  |  uzd: Δ vs uzdevuma labākais  |  lab: Δ vs {best_overall_disp}",
        fontsize=9,
        pad=8,
        loc="left",
    )

    cmap = LinearSegmentedColormap.from_list("perf_red", [(1, 1, 1), (1, 0.45, 0.45)])
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-80, vmax=-5))
    sm.set_array([])

    cax = fig.add_axes([l + w + 0.012, b, 0.013, h])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("% zem uzdevuma labākā", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.savefig(fn, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Uzdevumu-modeļu rangu tabula saglabāta: '{fn}'")


def plot_task_model_abs_table(long, fn):
    # ── Task + model ordering (same as ranking table) ────────────────────────
    task_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .sort_values(["domain", "task_name"])
    )
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]

    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    pivot = pivot.loc[ordered_tasks]

    best_overall = _find_best_overall_model(pivot)
    win_count = int((pivot.idxmax(axis=1) == best_overall).sum())
    best_overall_disp = model_short_name(best_overall)
    best_per_task = pivot.max(axis=1)

    n_tasks = len(ordered_tasks)
    n_models = pivot.shape[1]

    # Per-row ordering: worst → best on that task
    ranked_models = [
        pivot.loc[task].sort_values(na_position="first").index.tolist()
        for task in ordered_tasks
    ]

    score_mat = np.full((n_tasks, n_models), np.nan)
    model_mat = [["" for _ in range(n_models)] for _ in range(n_tasks)]
    diff_task_mat = np.full((n_tasks, n_models), np.nan)  # raw nDCG diff vs task-best
    diff_best_mat = np.full(
        (n_tasks, n_models), np.nan
    )  # raw nDCG diff vs overall-best

    for i, task in enumerate(ordered_tasks):
        best_score = best_per_task.iloc[i]
        best_overall_score = (
            pivot.loc[task, best_overall] if best_overall in pivot.columns else np.nan
        )
        for j, model in enumerate(ranked_models[i]):
            score = pivot.loc[task, model]
            score_mat[i, j] = score
            model_mat[i][j] = model_short_name(model)
            if not (np.isnan(score) or np.isnan(best_score)):
                diff_task_mat[i, j] = score - best_score
            if not (np.isnan(score) or np.isnan(best_overall_score)):
                diff_best_mat[i, j] = score - best_overall_score

    # Figure layout
    cell_w = 1.6
    cell_h = 0.85
    label_left = 5.2
    label_bottom = 0.8
    colorbar_right = 1.3

    fig_w = label_left + n_models * cell_w + colorbar_right
    fig_h = label_bottom + n_tasks * cell_h + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h))

    l = label_left / fig_w
    b = label_bottom / fig_h
    w = (n_models * cell_w) / fig_w
    h = (n_tasks * cell_h) / fig_h
    ax = fig.add_axes([l, b, w, h])

    ax.set_xlim(0, n_models)
    ax.set_ylim(0, n_tasks)
    ax.invert_yaxis()

    # Cell draw
    for i in range(n_tasks):
        for j in range(n_models):
            score = score_mat[i, j]
            d_task = diff_task_mat[i, j]
            d_best = diff_best_mat[i, j]
            mname = model_mat[i][j]

            if np.isnan(score):
                color = (0.88, 0.88, 0.88)
            else:
                t = np.clip(-d_task, 0.0, 1.0)  # 0 at task-best, 1 at diff=-1
                color = (1.0, 1.0 - t * 0.55, 1.0 - t * 0.55)

            rect = Rectangle((j, i), 1, 1, fc=color, ec="#b8b8b8", lw=0.5, zorder=1)
            ax.add_patch(rect)

            if not np.isnan(score):
                cell_txt = (
                    f"{_wrap_hyphens(mname)}\n"
                    f"{score:.3f}\n"
                    f"uzd:{d_task:+.3f}\n"
                    f"lab:{d_best:+.3f}"
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    cell_txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    family="monospace",
                    zorder=2,
                )

    ax.set_xticks([j + 0.5 for j in range(n_models)])
    ax.set_xticklabels([f"vieta {j + 1}" for j in range(n_models)], fontsize=8)
    ax.set_xlabel(
        "← sliktākais uzdevumā          labākais uzdevumā →", fontsize=8, labelpad=4
    )
    ax.set_yticks([i + 0.5 for i in range(n_tasks)])
    ax.set_yticklabels(task_labels, fontsize=7.5)

    ax.set_title(
        f"Uzdevums × Modelis  —  absolūtās vērtības  |  kolonnas = vieta uzdevumā  |  rindas = uzdevumi pēc domēna\n"
        f"Kopumā labākais modelis: {best_overall_disp}  (uzvar {win_count}/{n_tasks} uzdevumos)\n"
        f"Šūna: modeļa nosaukums, nDCG@10  |  uzd: nDCG starpība vs uzdevuma labākais  |  lab: nDCG starpība vs {best_overall_disp}",
        fontsize=9,
        pad=8,
        loc="left",
    )

    cmap = LinearSegmentedColormap.from_list("abs_red", [(1, 1, 1), (1, 0.45, 0.45)])
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=0))
    sm.set_array([])
    cax = fig.add_axes([l + w + 0.012, b, 0.013, h])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("nDCG starpība vs uzdevuma labākais", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.savefig(fn, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Uzdevumu-modeļu absolūto vērtību tabula saglabāta: '{fn}'")


def run(gm, suffix):
    df = get_scores_dataframe(gm)
    long = build_long_df(df)
    plot_task_model_ranking_table(
        long, thesis_file(f"task_model_ranking_table{suffix}.png")
    )
    plot_task_model_abs_table(long, thesis_file(f"task_model_abs_table{suffix}.png"))


def main():
    run(get_models, "")


if __name__ == "__main__":
    main()
