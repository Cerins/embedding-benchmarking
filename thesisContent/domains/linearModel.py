import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from common.utils import *


# Build the dataframe which matches the expected df
def build_long_df(df):
    # From 2d array where each task has model results to a row which is task x model
    model_cols = [c for c in df.columns if c not in ("task_name", "domains")]
    # Use model to take the model scores and transform into desired format
    long = df.melt(
        id_vars=["task_name", "domains"],
        value_vars=model_cols,
        var_name="model",
        value_name="score",
    )
    # Drop tasks where any model has a missing score
    complete_tasks = long.groupby("task_name")["score"].apply(lambda s: s.notna().all())
    long = long[long["task_name"].isin(complete_tasks[complete_tasks].index)]
    # Extract the single TARGET_DOMAINS domain from each task's domain list
    # There is a guarantee that there is only 1 TARGET_DOMAIN here
    long["domain"] = long["domains"].apply(
        lambda d: next(x for x in d if x in TARGET_DOMAINS)
    )
    # Make the model graphs more readable
    long["model_short"] = long["model"].apply(model_short_name)
    return long


# Long datarame to anova model and table
def fit_and_anova(long):
    # Try to fit data in a linear model where variables are model, the domain and tupple of model and domain(model specialization)
    model = smf.ols(
        "score ~ C(model_short) + C(domain) + C(model_short):C(domain)", data=long
    ).fit()
    # Get the statistics
    table = anova_lm(model, typ=2)
    return model, table


# Plot the effect of vizualation
def plot_interaction_effects(long, fn):
    pivot = long.groupby(["model_short", "domain"])["score"].mean().unstack()
    # Subtract each model's row mean so we see domain-relative performance
    pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
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
                ax.text(j, i, f"{val:+.3f}\n({abs_val:.3f})", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interaction effects plot saved to '{fn}'")


def plot_boxplot_models(long, fn):
    models = long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    data = [long.loc[long["model_short"] == m, "score"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(models))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels(models)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model box plot saved to '{fn}'")


def plot_boxplot_domains(long, fn):
    domains = long.groupby("domain")["score"].mean().sort_values().index.tolist()
    data = [long.loc[long["domain"] == d, "score"].dropna().values for d in domains]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(domains))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(domains) + 1))
    ax.set_yticklabels(domains)
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
                ax.text(j, i, f"{val:+.3f}\n({abs_val:.3f})", ha="center", va="center", fontsize=6)


def plot_interaction_effects_tasks(long, fn_prefix):
    # Build task labels: "task_name (domain)", sorted by domain then task name
    task_domain = long[["task_name", "domain"]].drop_duplicates().sort_values(["domain", "task_name"])
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]

    # Models sorted by mean performance (best last = top of y-axis equivalent on x)
    model_order = long.groupby("model_short")["score"].mean().sort_values().index.tolist()

    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    pivot = pivot.loc[ordered_tasks, model_order]
    pivot.index = task_labels
    pivot_centered = pivot.sub(pivot.mean(axis=0), axis=1)

    mid = len(ordered_tasks) // 2
    for part, (pc_part, pv_part, suffix) in enumerate(
        [
            (pivot_centered.iloc[:mid], pivot.iloc[:mid], "1"),
            (pivot_centered.iloc[mid:], pivot.iloc[mid:], "2"),
        ]
    ):
        n_tasks = len(pc_part)
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(model_order)), max(4, 0.45 * n_tasks)))
        _render_task_interaction(pc_part, pv_part, ax)
        plt.tight_layout()
        fn = f"{fn_prefix}_{suffix}.png"
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Task interaction effects plot saved to '{fn}'")


def plot_anova_variance(anova_table, fn):
    # Anova dispersiju izskaidrošanas pamats
    rows = anova_table[anova_table.index != "Residual"].copy()
    rows["pct"] = rows["sum_sq"] / anova_table["sum_sq"].sum() * 100
    residual_pct = (
        anova_table.loc["Residual", "sum_sq"] / anova_table["sum_sq"].sum() * 100
    )
    labels = list(rows.index) + ["Residual"]
    values = list(rows["pct"]) + [residual_pct]
    colors = ["C0", "C1", "C2", "C3"]
    fig, ax = plt.subplots(figsize=(7, 4))
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


def main():
    df = get_scores_dataframe(TARGET_DOMAINS)
    long = build_long_df(df)
    print(f"Observations: {len(long)}")
    print(long[["domain", "model_short", "score"]].describe())
    model, anova_table = fit_and_anova(long)
    print("Model Summary")
    print(model.summary())
    print("ANOVA table")
    print(anova_table.to_string())
    # Save ANOVA table as CSV for reference
    anova_table.to_csv(thesis_file("domain_lm_anova.csv"))
    plot_interaction_effects(long, thesis_file("domain_lm_interaction_effects.png"))
    plot_interaction_effects_tasks(long, thesis_file("domain_lm_interaction_effects_tasks"))
    plot_anova_variance(anova_table, thesis_file("domain_lm_anova_variance.png"))
    plot_boxplot_models(long, thesis_file("domain_lm_boxplot_models.png"))
    plot_boxplot_domains(long, thesis_file("domain_lm_boxplot_domains.png"))


if __name__ == "__main__":
    main()
