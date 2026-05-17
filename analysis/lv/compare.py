import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from common.utils import (
    thesis_file,
    mteb_to_hf_repo,
    get_latvian_models,
    _scan_scores_dir,
    _extract_ndcg_at_10,
)
from common.style import PALETTE

TASKS = ["WikiLV", "MultiEupV2LV"]

_COMPUTED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "computed")


def load_scores(task_names):
    task_set = set(task_names)
    model_scores = defaultdict(dict)
    _scan_scores_dir(_COMPUTED_DIR, task_set, _extract_ndcg_at_10, model_scores, None)
    return {
        m: dict(scores)
        for m, scores in model_scores.items()
        if all(t in scores for t in task_names)
    }


def plot(model_scores, task_names, fn, highlight=None):
    means = {m: np.mean([s[t] for t in task_names]) for m, s in model_scores.items()}
    ordered = sorted(means, key=means.get, reverse=True)
    best_avg = means[ordered[0]]

    # reverse so best ends up at the top (barh plots bottom-to-top)
    plot_order = list(reversed(ordered))
    labels = [mteb_to_hf_repo(m) for m in plot_order]

    colors = PALETTE
    n_tasks = len(task_names)
    slot = 0.8 / n_tasks
    y = np.arange(len(plot_order))

    xlim = (0, min(1.0, best_avg + 0.25))

    fig, (ax_bar, ax_box) = plt.subplots(
        2,
        1,
        figsize=(14, max(6, len(plot_order) * 0.55) + 2),
        gridspec_kw={"height_ratios": [len(plot_order), n_tasks]},
        sharex=True,
    )

    # grouped horizontal bar chart: one bar per task per model
    for ti, task in enumerate(task_names):
        offsets = (ti - (n_tasks - 1) / 2) * slot
        task_values = [model_scores[m][task] for m in plot_order]
        best_task = max(task_values)
        bars = ax_bar.barh(y + offsets, task_values, slot, label=task, color=colors[ti % len(colors)])
        for bar_i, (bar, val) in enumerate(zip(bars, task_values)):
            model = plot_order[bar_i]
            is_best = model == ordered[0]
            cy = bar.get_y() + bar.get_height() / 2
            label = f"{val:.3f}" if is_best else f"{val:.3f} ({val/best_task*100:.1f}%)"
            ax_bar.text(val + 0.002, cy, label, ha="left", va="center", fontsize=6)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=8)
    for tick, model in zip(ax_bar.get_yticklabels(), plot_order):
        if model == highlight:
            tick.set_fontweight("bold")
    ax_bar.set_xlim(*xlim)
    ax_bar.legend(title="Task", fontsize=8)

    # horizontal box and whisker per task, sharing x-axis with bar chart
    box_data = [[model_scores[m][t] for m in ordered] for t in task_names]
    ax_box.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        boxprops=dict(alpha=0.6),
    )
    ax_box.set_yticks(range(1, n_tasks + 1))
    ax_box.set_yticklabels(task_names, fontsize=8)
    ax_box.set_xlabel("nDCG@10")
    ax_box.set_xlim(*xlim)

    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to '{fn}'")


def main():
    print(f"Tasks: {TASKS}")
    latvian_models = set(get_latvian_models())
    model_scores = {
        m: s for m, s in load_scores(TASKS).items() if m in latvian_models
    }
    print(f"Models with full coverage: {len(model_scores)}")
    plot(model_scores, TASKS, thesis_file("lv_compare.png"), highlight="lv-mbert-embed-base")


if __name__ == "__main__":
    main()
