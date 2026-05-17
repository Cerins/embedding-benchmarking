import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from common.utils import *
from common.ranking import kendall_tau, compute_distance_matrix
from analysis.domains.anova import build_long_df


# MDS plotting for tasks
def plot_mds(pivot, task_to_domain, fn):
    tasks = [t for t in pivot.index if t in task_to_domain]
    if len(tasks) < 2:
        print(f"not enough tasks to plot ({len(tasks)})")
        return

    pivot_kept = pivot.loc[tasks]
    D = compute_distance_matrix(pivot_kept)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        n_init=8,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(D)
    normalized_stress = np.sqrt(mds.stress_ / (0.5 * np.sum(D**2)))
    print(f"  MDS stress: {mds.stress_:.4f}  normalized: {normalized_stress:.4f}")

    domains = sorted({task_to_domain[t] for t in tasks})
    cmap = plt.cm.tab20 if len(domains) <= 20 else plt.cm.gist_ncar
    palette = cmap(np.linspace(0, 1, max(len(domains), 1)))
    domain_to_color = {d: palette[i] for i, d in enumerate(domains)}

    fig, ax = plt.subplots(figsize=(15, 11))
    for (x, y), t in zip(coords, tasks):
        domain = task_to_domain[t]
        color = domain_to_color[domain]
        ax.scatter(
            x,
            y,
            color=color,
            marker="o",
            s=80,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.annotate(
            f"{t}|{domain}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6.5,
            zorder=4,
        )

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(
        "Uzdevumi MDS telpā pēc modeļu rangiem\n(distance = (1 − Kendall τ) / 2)",
        fontsize=11,
    )
    ax.axhline(0, color="grey", lw=0.4, linestyle="--")
    ax.axvline(0, color="grey", lw=0.4, linestyle="--")
    ax.grid(True, alpha=0.2)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=domain_to_color[d],
            markeredgecolor="black",
            markersize=8,
            label=d,
        )
        for d in domains
    ]
    ncol = max(1, len(handles) // 14 + 1)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        ncol=ncol,
        title="Domēns",
    )

    plt.tight_layout()
    plt.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fn}")


def run(gm, suffix):
    df = get_scores_dataframe(gm)
    long = build_long_df(df)
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    task_to_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .set_index("task_name")["domain"]
        .to_dict()
    )
    plot_mds(pivot, task_to_domain, thesis_file(f"task_kendall_tau_mds{suffix}.png"))


def main():
    run(get_models, "")


if __name__ == "__main__":
    main()
