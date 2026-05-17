import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from common.utils import *


def build_long_df(df):
    # From 2d array where each task has model results to a row which is task x model
    model_cols = [c for c in df.columns if c not in ("task_name", "domains")]
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
    long["domain"] = long["domains"].apply(
        lambda d: next(x for x in d if x in TARGET_DOMAINS)
    )
    long["model_short"] = long["model"].apply(mteb_to_hf_repo)
    return long


# Render heatmap with deviation, absolute value, and % deviation per cell
def _render_interaction(pivot_centered, pivot, label, ax):
    im = ax.imshow(pivot_centered.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot_centered.columns)))
    ax.set_xticklabels(pivot_centered.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_centered.index)))
    ax.set_yticklabels(pivot_centered.index)
    plt.colorbar(im, ax=ax, label=label)
    for i in range(len(pivot_centered.index)):
        for j in range(len(pivot_centered.columns)):
            val = pivot_centered.values[i, j]
            abs_val = pivot.values[i, j]
            if not np.isnan(val):
                # val = abs_val - centering_mean, so mean = abs_val - val
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


# center_by: "model" subtracts each model's mean (model-relative deviation);
#            "task" subtracts each task's mean (task-relative deviation).
def plot_interaction_effects(long, fn, center_by, models_on_y=False):
    task_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .sort_values(["domain", "task_name"])
    )
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]
    model_order = (
        long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    )

    if models_on_y:
        pivot = long.groupby(["model_short", "task_name"])["score"].mean().unstack()
        pivot = pivot.loc[model_order, ordered_tasks]
        pivot.columns = task_labels
        figsize = (max(8, 1.4 * len(ordered_tasks)), max(4, 0.9 * len(model_order)))
        rows_are_models = True
    else:
        pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
        pivot = pivot.loc[ordered_tasks, model_order]
        pivot.index = task_labels
        figsize = (max(8, 0.9 * len(model_order)), max(4, 0.45 * len(ordered_tasks)))
        rows_are_models = False

    # Subtract the mean along the axis matching center_by
    center_on_rows = (center_by == "model") == rows_are_models
    if center_on_rows:
        pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0)
    else:
        pivot_centered = pivot.sub(pivot.mean(axis=0), axis=1)

    label = (
        "nDCG@10 (novirze no modeļa vidējā)"
        if center_by == "model"
        else "nDCG@10 (novirze no uzdevuma vidējā)"
    )

    fig, ax = plt.subplots(figsize=figsize)
    _render_interaction(pivot_centered, pivot, label, ax)
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interaction effects ({center_by}-relative) plot saved to '{fn}'")


# Linear model: model + domain + task-nested-in-domain + model:domain interaction.
# Each task belongs to exactly one domain, so task is nested (not crossed) inside
# domain. Writing it as C(domain):C(task_name) keeps the SS partition honest:
# domain captures between-domain variance and the nested term captures
# between-task-within-domain variance, instead of task absorbing all of domain.
TERM_LABELS_LV = {
    "C(model_short)": "Modelis",
    "C(domain)": "Nozare",
    "C(model_short):C(domain)": "Modelis x Nozare",
    "C(domain):C(task_name)": "Uzdevums",
    "C(task_name)": "Uzdevums",
    "Residual": "Atlikums",
}


def fit_and_anova(long):
    model = smf.ols(
        # "score ~ C(model_short) + C(domain) + C(domain):C(task_name)",
        "score ~ C(model_short) + C(domain) + C(task_name)",
        data=long,
    ).fit()
    table2 = anova_lm(model, typ=2)
    return model, table2


# Plot the anova variance
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


def plot_boxplot_tasks(long, fn):
    means = long.groupby("task_name")["score"].mean()
    task_domain = long[["task_name", "domain"]].drop_duplicates()
    task_domain = task_domain.assign(mean=task_domain["task_name"].map(means))
    # Sort ascending so best mean ends up at top of horizontal plot
    task_domain = task_domain.sort_values("mean", ascending=True)
    tasks = task_domain["task_name"].tolist()
    labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]

    data = [long.loc[long["task_name"] == t, "score"].dropna().values for t in tasks]
    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(tasks))))
    ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(tasks) + 1))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("nDCG@10")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Task box plot saved to '{fn}'")


def run(gm, suffix):
    df = get_scores_dataframe(gm).dropna()
    # Transform df into proper
    long = build_long_df(df)
    print(f"Observations: {len(long)}")
    model, anova_table2 = fit_and_anova(long)
    print("Model Summary")
    print(model.summary())
    print("ANOVA table (Type II)")
    print(anova_table2.to_string())
    anova_table2.to_csv(thesis_file(f"domain_task_lm_anova{suffix}.csv"))
    # Plot variance
    plot_anova_variance(
        anova_table2, thesis_file(f"domain_task_lm_anova_variance{suffix}.png")
    )
    # Plot boxplots
    plot_boxplot_tasks(long, thesis_file(f"domain_task_lm_boxplot_tasks{suffix}.png"))
    models_on_y = suffix == "_full"
    # Each cell shown as deviation from that model's mean
    plot_interaction_effects(
        long,
        thesis_file(f"domain_task_lm_interaction_effects_tasks{suffix}.png"),
        center_by="model",
        models_on_y=models_on_y,
    )
    # Each cell shown as deviation from that task's mean
    plot_interaction_effects(
        long,
        thesis_file(f"domain_task_lm_task_relative_effects_tasks{suffix}.png"),
        center_by="task",
        models_on_y=models_on_y,
    )


def main():
    run(get_models, "")
    # run(get_multilingual_models, "_full")


if __name__ == "__main__":
    main()
