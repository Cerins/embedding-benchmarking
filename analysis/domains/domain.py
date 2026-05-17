import numpy as np
import matplotlib.pyplot as plt
from common.utils import *


def build_long_df(df):
    model_cols = [c for c in df.columns if c not in ("task_name", "domains")]
    long = df.melt(
        id_vars=["task_name", "domains"],
        value_vars=model_cols,
        var_name="model",
        value_name="score",
    )
    complete_tasks = long.groupby("task_name")["score"].apply(lambda s: s.notna().all())
    long = long[long["task_name"].isin(complete_tasks[complete_tasks].index)]
    # There is a guarantee that there is only 1 TARGET_DOMAIN per task
    long["domain"] = long["domains"].apply(
        lambda d: next(x for x in d if x in TARGET_DOMAINS)
    )
    long["model_short"] = long["model"].apply(mteb_to_hf_repo)
    return long


def plot_boxplot_models(long, fn):
    models = long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    task_counts = long.groupby("model_short")["task_name"].nunique()
    data = [long.loc[long["model_short"] == m, "score"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(models))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels([f"{m} ({task_counts[m]})" for m in models], fontsize=7)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model box plot saved to '{fn}'")


def plot_boxplot_domains(long, fn):
    domains = long.groupby("domain")["score"].mean().sort_values().index.tolist()
    task_counts = long.groupby("domain")["task_name"].nunique()
    data = [long.loc[long["domain"] == d, "score"].dropna().values for d in domains]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(domains))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(domains) + 1))
    ax.set_yticklabels([f"{d} ({task_counts[d]})" for d in domains], fontsize=7)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Domain box plot saved to '{fn}'")


def _render_task_interaction(pivot_centered, pivot, ax):
    im = ax.imshow(pivot_centered.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot_centered.columns)))
    ax.set_xticklabels(pivot_centered.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_centered.index)))
    ax.set_yticklabels(pivot_centered.index)
    plt.colorbar(im, ax=ax, label="nDCG@10 (novirze no modeļa vidējā)")
    for i in range(len(pivot_centered.index)):
        for j in range(len(pivot_centered.columns)):
            val = pivot_centered.values[i, j]
            abs_val = pivot.values[i, j]
            if not np.isnan(val):
                mean = abs_val - val
                pct = (val / mean * 100) if mean != 0 else 0.0
                ax.text(j, i, f"{val:+.3f}\n({abs_val:.3f})\n{pct:+.1f}%", ha="center", va="center", fontsize=6)


def plot_interaction_effects_tasks(long, fn, models_on_y=False):
    task_domain = long[["task_name", "domain"]].drop_duplicates().sort_values(["domain", "task_name"])
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]

    model_order = long.groupby("model_short")["score"].mean().sort_values().index.tolist()

    if models_on_y:
        pivot = long.groupby(["model_short", "task_name"])["score"].mean().unstack()
        pivot = pivot.loc[model_order, ordered_tasks]
        pivot.columns = task_labels
        pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0)
        figsize = (max(8, 1.4 * len(ordered_tasks)), max(4, 0.9 * len(model_order)))
    else:
        pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
        pivot = pivot.loc[ordered_tasks, model_order]
        pivot.index = task_labels
        pivot_centered = pivot.sub(pivot.mean(axis=0), axis=1)
        figsize = (max(8, 0.9 * len(model_order)), max(4, 0.45 * len(ordered_tasks)))

    fig, ax = plt.subplots(figsize=figsize)
    _render_task_interaction(pivot_centered, pivot, ax)
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Task interaction effects plot saved to '{fn}'")


def run(gm, suffix):
    df = get_scores_dataframe(gm)
    long = build_long_df(df)
    models_on_y = suffix == "_full"
    plot_interaction_effects_tasks(long, thesis_file(f"domain_lm_interaction_effects_tasks{suffix}.png"), models_on_y=models_on_y)
    plot_boxplot_models(long, thesis_file(f"domain_lm_boxplot_models{suffix}.png"))
    plot_boxplot_domains(long, thesis_file(f"domain_lm_boxplot_domains{suffix}.png"))


def main():
    run(get_models, "")
    # run(get_multilingual_models, "_full")


if __name__ == "__main__":
    main()
