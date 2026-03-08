import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from common.utils import *

from thesisContent.domains.linearModel import build_long_df


def fit_and_anova(long):
    # Linear model with model, domain, task and model-domain interaction
    model = smf.ols(
        "score ~ C(model_short) + C(domain) + C(task_name) + C(model_short):C(domain)",
        data=long,
    ).fit()
    table = anova_lm(model, typ=2)
    return model, table


def plot_anova_variance(anova_table, fn):
    rows = anova_table[anova_table.index != "Residual"].copy()
    total_ss = anova_table["sum_sq"].sum()
    rows["pct"] = rows["sum_sq"] / total_ss * 100
    residual_pct = anova_table.loc["Residual", "sum_sq"] / total_ss * 100
    labels = list(rows.index) + ["Residual"]
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


def plot_boxplot_tasks(long, fn_prefix):
    means = long.groupby("task_name")["score"].mean()
    task_domain = long[["task_name", "domain"]].drop_duplicates()
    task_domain = task_domain.assign(mean=task_domain["task_name"].map(means))
    # Sort ascending so best mean ends up at top of horizontal plot
    task_domain = task_domain.sort_values("mean", ascending=True)
    tasks = task_domain["task_name"].tolist()
    labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]

    mid = len(tasks) // 2
    for items, suffix in [((tasks[:mid], labels[:mid]), "1"), ((tasks[mid:], labels[mid:]), "2")]:
        t_slice, l_slice = items
        data = [long.loc[long["task_name"] == t, "score"].dropna().values for t in t_slice]
        fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(t_slice))))
        ax.boxplot(data, vert=False, patch_artist=True)
        ax.set_yticks(range(1, len(t_slice) + 1))
        ax.set_yticklabels(l_slice, fontsize=7)
        ax.set_xlabel("nDCG@10")
        plt.tight_layout()
        fn = f"{fn_prefix}_{suffix}.png"
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Task box plot saved to '{fn}'")


def main():
    df = get_scores_dataframe(TARGET_DOMAINS)
    long = build_long_df(df)
    print(f"Observations: {len(long)}")
    model, anova_table = fit_and_anova(long)
    print("Model Summary")
    print(model.summary())
    print("ANOVA table")
    print(anova_table.to_string())
    anova_table.to_csv(thesis_file("domain_task_lm_anova.csv"))
    plot_anova_variance(anova_table, thesis_file("domain_task_lm_anova_variance.png"))
    plot_boxplot_tasks(long, thesis_file("domain_task_lm_boxplot_tasks"))


if __name__ == "__main__":
    main()
