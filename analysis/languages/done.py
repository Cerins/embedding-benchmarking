import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from common.utils import *
from common.style import CoverageColor

SOURCE_TO_INT = {
    ScoreSource.CACHE: 1,
    ScoreSource.COMPUTED: 2,
}


def df_to_matrix(df):
    task_names = list(df["task_name"])
    models = [c for c in df.columns if "__" in c]
    matrix = np.zeros((len(task_names), len(models)), dtype=int)
    for j, model in enumerate(models):
        for i, val in enumerate(df[model]):
            matrix[i, j] = SOURCE_TO_INT.get(val, 0)
    return task_names, models, matrix


def visualize_grid(df, output_file):
    task_names, models, matrix = df_to_matrix(df)
    cmap = ListedColormap(
        [
            CoverageColor.MISSING.value,
            CoverageColor.CACHE.value,
            CoverageColor.COMPUTED.value,
        ]
    )
    fig, ax = plt.subplots(figsize=(len(models) * 0.45 + 6, len(task_names) * 0.28 + 6))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_yticklabels(task_names, fontsize=7)
    ax.set_xlabel("Modeļi")
    ax.set_ylabel("Uzdevumi")
    legend = []
    if np.any(matrix == 0):
        legend.append(
            Patch(facecolor=CoverageColor.MISSING.value, label="Netika apstrādāts")
        )
    legend.extend(
        [
            Patch(
                facecolor=CoverageColor.CACHE.value,
                label="Globālas kešatmiņas rezultāts",
            ),
            Patch(
                facecolor=CoverageColor.COMPUTED.value,
                label="Manuāli aprēķināts rezultāts",
            ),
        ]
    )
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(thesis_file(output_file), dpi=300)
    plt.close()
    print("Saved:", output_file)


def save_stats(df, output_file):
    task_names, models, matrix = df_to_matrix(df)
    stats = {
        "total_tasks": len(task_names),
        "total_models": len(models),
        "total_computed_cells": int(
            np.sum(matrix == SOURCE_TO_INT[ScoreSource.COMPUTED])
        ),
        "total_cached_cells": int(np.sum(matrix == SOURCE_TO_INT[ScoreSource.CACHE])),
    }
    with open(thesis_file(output_file), "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved stats:", output_file)
    return stats


def main():
    print("Loading multilingual benchmark...")
    df = get_benchmark_dataframe(get_multilingual_models, multilingual=True).dropna()
    visualize_grid(df, "lang_model_dataset_done.png")
    save_stats(df, "lang_model_dataset_stats.json")


if __name__ == "__main__":
    main()
