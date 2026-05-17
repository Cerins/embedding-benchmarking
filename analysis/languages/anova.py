import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from common.utils import *


def build_long_df(df):
    model_cols = [
        c for c in df.columns if c not in ("task_name", "domains", "languages")
    ]
    long = df.melt(
        id_vars=["task_name", "domains", "languages"],
        value_vars=model_cols,
        var_name="model",
        value_name="score",
    )
    # Drop tasks where any model has a missing score
    complete_tasks = long.groupby("task_name")["score"].apply(lambda s: s.notna().all())
    long = long[long["task_name"].isin(complete_tasks[complete_tasks].index)]
    long["model_short"] = long["model"].apply(mteb_to_hf_repo)
    return long


TERM_LABELS_LV = {
    "C(model_short)": "Modelis",
    "C(task_name)": "Uzdevums",
    "Residual": "Atlikums",
}


def fit_and_anova(long):
    model = smf.ols(
        "score ~ C(model_short) + C(task_name)",
        data=long,
    ).fit()
    table2 = anova_lm(model, typ=2)
    return model, table2


# Plot language variance
def plot_anova_variance(anova_table, fn):
    rows = anova_table[anova_table.index != "Residual"].copy()
    total_ss = anova_table["sum_sq"].sum()
    rows["pct"] = rows["sum_sq"] / total_ss * 100
    residual_pct = anova_table.loc["Residual", "sum_sq"] / total_ss * 100
    raw_labels = list(rows.index) + ["Residual"]
    labels = [TERM_LABELS_LV.get(l, l) for l in raw_labels]
    values = list(rows["pct"]) + [residual_pct]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylabel("% no kopējās dispersijas (SS)")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ANOVA variance plot saved to '{fn}'")


# Boxplots
def plot_boxplot_models(long, fn):
    models = long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    data = [long.loc[long["model_short"] == m, "score"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(models))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model box plot saved to '{fn}'")


def plot_boxplot_tasks(long, fn):
    means = long.groupby("task_name")["score"].mean()
    task_df = long[["task_name"]].drop_duplicates()
    task_df = task_df.assign(mean=task_df["task_name"].map(means))
    task_df = task_df.sort_values("mean", ascending=True)
    tasks = task_df["task_name"].tolist()

    data = [long.loc[long["task_name"] == t, "score"].dropna().values for t in tasks]
    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(tasks))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(tasks) + 1))
    ax.set_yticklabels(tasks, fontsize=7)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Task box plot saved to '{fn}'")


def plot_interaction_effects(long, fn):
    pivot = long.groupby(["model_short", "task_name"])["score"].mean().unstack()
    pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0)
    n_tasks = len(pivot_centered.columns)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * n_tasks), max(4, 0.45 * len(pivot_centered.index)))
    )
    im = ax.imshow(pivot_centered.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(pivot_centered.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot_centered.index)))
    ax.set_yticklabels(pivot_centered.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="nDCG@10 (novirze no modeļa vidējā)")
    for i in range(len(pivot_centered.index)):
        for j in range(n_tasks):
            val = pivot_centered.values[i, j]
            abs_val = pivot.values[i, j]
            if not np.isnan(val):
                mean = abs_val - val
                pct = (val / mean * 100) if mean != 0 else 0.0
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}\n({abs_val:.3f})\n{pct:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interaction effects plot saved to '{fn}'")


def plot_task_relative_effects(long, fn):
    pivot = long.groupby(["model_short", "task_name"])["score"].mean().unstack()
    task_means = pivot.mean(axis=0)
    pivot_centered = pivot.sub(task_means, axis=1)
    n_tasks = len(pivot_centered.columns)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * n_tasks), max(4, 0.45 * len(pivot_centered.index)))
    )
    im = ax.imshow(pivot_centered.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(pivot_centered.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot_centered.index)))
    ax.set_yticklabels(pivot_centered.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="nDCG@10 (novirze no uzdevuma vidējā)")
    for i in range(len(pivot_centered.index)):
        for j in range(n_tasks):
            val = pivot_centered.values[i, j]
            abs_val = pivot.values[i, j]
            if not np.isnan(val):
                task_mean = task_means.iloc[j]
                pct = (val / task_mean * 100) if task_mean != 0 else 0.0
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}\n({abs_val:.3f})\n{pct:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Task-relative interaction effects plot saved to '{fn}'")


def run(gm, suffix):
    df = get_multilingual_scores_dataframe(gm)
    long = build_long_df(df)
    print(f"Observations: {len(long)}")
    model, anova_table2 = fit_and_anova(long)
    print("Model Summary")
    print(model.summary())
    print("ANOVA table (Type II)")
    print(anova_table2.to_string())
    anova_table2.to_csv(thesis_file(f"lang_task_lm_anova{suffix}.csv"))
    plot_anova_variance(
        anova_table2, thesis_file(f"lang_task_lm_anova_variance{suffix}.png")
    )
    plot_boxplot_models(long, thesis_file(f"lang_task_lm_boxplot_models{suffix}.png"))
    plot_boxplot_tasks(long, thesis_file(f"lang_task_lm_boxplot_tasks{suffix}.png"))
    plot_interaction_effects(
        long, thesis_file(f"lang_task_lm_interaction_effects{suffix}.png")
    )
    plot_task_relative_effects(
        long, thesis_file(f"lang_task_lm_task_relative_effects{suffix}.png")
    )


def main():
    run(get_multilingual_models, "")


if __name__ == "__main__":
    main()
