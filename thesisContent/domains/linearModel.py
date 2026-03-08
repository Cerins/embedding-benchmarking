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
    long["model_short"] = long["model"].apply(lambda m: m.split("__")[-1])
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
    plot_anova_variance(anova_table, thesis_file("domain_lm_anova_variance.png"))


if __name__ == "__main__":
    main()
