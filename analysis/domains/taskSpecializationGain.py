import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from common.utils import *
from analysis.domains.anova import build_long_df
from analysis.domains.modelRankingTable import _find_best_overall_model


# Gain is best task score - best overall mode score
def compute_gains(long):
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    best_overall = _find_best_overall_model(pivot)
    best_overall_disp = model_short_name(best_overall)
    win_count = int((pivot.idxmax(axis=1) == best_overall).sum())
    task_domain = long[["task_name", "domain"]].drop_duplicates().set_index("task_name")
    gains = []
    for task in pivot.index:
        row = pivot.loc[task].dropna()
        if row.empty or best_overall not in row.index:
            continue
        best_task_score = row.max()
        best_task_model = row.idxmax()
        overall_score = row[best_overall]
        gain = best_task_score - overall_score
        gains.append(
            {
                "task_name": task,
                "domain": task_domain.loc[task, "domain"],
                "gain": gain,
                "best_task_model": model_short_name(best_task_model),
                "best_task_score": best_task_score,
                "overall_score": overall_score,
            }
        )
    df = pd.DataFrame(gains).sort_values("gain")
    return df, best_overall_disp, win_count


def plot_gain_distribution(df, best_overall_disp, win_count, fn):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.28 * len(df))))
    domains = sorted(df["domain"].unique())
    domain_colors = {
        d: plt.cm.tab10(i / max(len(domains) - 1, 1)) for i, d in enumerate(domains)
    }

    ax = axes[0]
    colors = [domain_colors[d] for d in df["domain"]]
    bars = ax.barh(
        range(len(df)), df["gain"], color=colors, edgecolor="none", height=0.7
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{r.task_name} ({r.domain})" for r in df.itertuples()],
        fontsize=6.5,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(
        df["gain"].mean(),
        color="crimson",
        lw=1.2,
        linestyle="--",
        label=f"vidējais {df['gain'].mean():.3f}",
    )
    ax.axvline(
        df["gain"].median(),
        color="darkorange",
        lw=1.2,
        linestyle=":",
        label=f"mediāna {df['gain'].median():.3f}",
    )
    ax.set_xlabel("Absolūtais nDCG@10 ieguvums", fontsize=9)
    ax.set_title(
        f"Uzdevumu labākā modeļa ieguvums\npret {best_overall_disp} ({win_count} uzvaras)",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    for d, c in domain_colors.items():
        ax.bar(0, 0, color=c, label=d)
    ax.legend(fontsize=7, loc="lower right")

    ax2 = axes[1]
    gains = df["gain"].values
    ax2.hist(
        gains,
        bins=12,
        color="#c0c0c0",
        edgecolor="white",
        density=True,
        label="histogramma",
    )

    ax2.axvline(
        gains.mean(),
        color="crimson",
        lw=1.5,
        linestyle="--",
        label=f"vidējais {gains.mean():.3f}",
    )
    ax2.axvline(
        np.median(gains),
        color="darkorange",
        lw=1.5,
        linestyle=":",
        label=f"mediāna {np.median(gains):.3f}",
    )
    ax2.set_xlabel("Absolūtais nDCG@10 ieguvums", fontsize=9)
    ax2.set_ylabel("Blīvums", fontsize=9)
    ax2.set_title("Sadalījums", fontsize=9)
    ax2.legend(fontsize=8)

    pct_positive = (gains > 0).mean() * 100
    ax2.text(
        0.97,
        0.97,
        f"{pct_positive:.0f}% uzdevumu: uzdevumu labākais -> {best_overall_disp}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="dimgray",
    )

    # plt.suptitle(
    #     f"Labākais modelis uzdevumam pret labākais vispārīgais modelis ({best_overall_disp})",
    #     fontsize=10,
    #     y=1.01,
    # )
    plt.tight_layout()
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Specializācijas pieauguma sadalījums saglabāts: '{fn}'")


def run(gm, suffix):
    df_scores = get_scores_dataframe(gm)
    long = build_long_df(df_scores)
    gains_df, best_overall_disp, win_count = compute_gains(long)
    plot_gain_distribution(
        gains_df,
        best_overall_disp,
        win_count,
        thesis_file(f"task_specialization_gain{suffix}.png"),
    )


def main():
    run(get_models, "")


if __name__ == "__main__":
    main()
