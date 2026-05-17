import numpy as np
import matplotlib.pyplot as plt

from common.utils import (
    thesis_file,
    get_eng_tasks,
    get_task_domains,
    model_short_name,
    _build_scores_df,
    _extract_ndcg_at_10,
)
from common.style import PALETTE

MODEL_CODE = "jinaai__jina-code-embeddings-0.5b"
MODEL_SMALL = "jinaai__jina-embeddings-v5-text-small"


def get_programming_tasks():
    tasks = get_eng_tasks(domain_filter=False)
    return [t for t in tasks if "programming" in get_task_domains(t)]


def plot_comparison(tasks, scores_code, scores_small, fn):
    shared = sorted([t for t in tasks if t in scores_code and t in scores_small])

    x = np.arange(len(shared))
    bar_w = 0.35

    vals_code = np.array([scores_code[t] for t in shared])
    vals_small = np.array([scores_small[t] for t in shared])

    fig, ax = plt.subplots(figsize=(max(14, len(shared) * 1.4), 7))

    bars_code = ax.bar(x - bar_w / 2, vals_code, bar_w, label=model_short_name(MODEL_CODE), color=PALETTE[0])
    bars_small = ax.bar(x + bar_w / 2, vals_small, bar_w, label=model_short_name(MODEL_SMALL), color=PALETTE[1])

    for bar, val in zip(bars_code, vals_code):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val / 2,
            f"{val:.3f}",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
            rotation=90,
        )
    for bar, val in zip(bars_small, vals_small):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val / 2,
            f"{val:.3f}",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
            rotation=90,
        )

    for i, (vc, vs) in enumerate(zip(vals_code, vals_small)):
        diff = vc - vs
        pct = (diff / vs * 100) if vs != 0 else 0.0
        top = max(vc, vs) + 0.012
        sign = "+" if diff >= 0 else ""
        ax.text(
            x[i],
            top,
            f"{sign}{diff:.3f}\n({sign}{pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(shared, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, min(1.0, max(max(vals_code), max(vals_small)) + 0.12))
    ax.legend()

    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to '{fn}'")


def main():
    tasks = get_programming_tasks()
    print(f"Programming tasks: {len(tasks)}")

    df = _build_scores_df(
        tasks=tasks,
        qualified_models=[MODEL_CODE, MODEL_SMALL],
        score_extractor=_extract_ndcg_at_10,
        drop_empty_rows=True,
    )

    scores_code = dict(zip(df["task_name"], df[MODEL_CODE]))
    scores_small = dict(zip(df["task_name"], df[MODEL_SMALL]))

    print(f"  {model_short_name(MODEL_CODE)}: {len(scores_code)} tasks")
    print(f"  {model_short_name(MODEL_SMALL)}: {len(scores_small)} tasks")

    plot_comparison(
        list(df["task_name"]),
        scores_code,
        scores_small,
        thesis_file("code_jina_compare.png"),
    )


if __name__ == "__main__":
    main()
