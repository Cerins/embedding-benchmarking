import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
import mteb
from common.utils import *

mteb_tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])

task_map = {t.metadata.name: t for t in mteb_tasks}


def visualize_grid(df, output_file):
    task_names = list(df["task_name"])
    models = list(df.columns[2:])
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    matrix = np.zeros((len(task_names), len(models)), dtype=int)
    for j, model in enumerate(models):
        model_dir = os.path.join(results_dir, model)
        if not os.path.exists(model_dir):
            continue
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                if file == "model_meta.json":
                    continue
                task = file[:-5]
                if task not in task_names:
                    continue
                i = task_names.index(task)
                path = os.path.join(root, file)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("our_test", False):
                        matrix[i, j] = 2
                    else:
                        matrix[i, j] = 1
                except:
                    pass

    cmap = ListedColormap(
        [
            "#cccccc",  # missing
            "#ffd92f",  # cached
            "#4daf4a",  # our_test
        ]
    )
    fig, ax = plt.subplots(figsize=(len(models) * 0.45 + 6, len(task_names) * 0.28 + 6))
    ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_yticklabels(task_names, fontsize=7)
    ax.set_xlabel("Models")
    ax.set_ylabel("Datasets")
    ax.set_title("Model vs Dataset Evaluation Coverage")

    for i, task in enumerate(task_names):
        task_obj = task_map.get(task)
        if task_obj and dataset_too_large(task_obj):
            rect = Rectangle(
                (-0.5, i - 0.5),
                len(models),
                1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
    legend = [
        Patch(facecolor="#cccccc", label="Netika apstrādāts"),
        Patch(facecolor="#ffd92f", label="Globālas kešatmiņas rezultāts"),
        Patch(facecolor="#4daf4a", label="Manuāli aprēķināts rezultāts"),
        Patch(
            facecolor="none",
            edgecolor="red",
            linewidth=2,
            label="Datu kopa pārāk liela",
        ),
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(thesis_file(output_file), dpi=300)
    plt.close()
    print("Saved:", output_file)


def save_stats(df, output_file):
    task_names = list(df["task_name"])
    models = list(df.columns[2:])
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    matrix = np.zeros((len(task_names), len(models)), dtype=int)
    for j, model in enumerate(models):
        model_dir = os.path.join(results_dir, model)
        if not os.path.exists(model_dir):
            continue
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if not file.endswith(".json") or file == "model_meta.json":
                    continue
                task = file[:-5]
                if task not in task_names:
                    continue
                i = task_names.index(task)
                path = os.path.join(root, file)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("our_test", False):
                        matrix[i, j] = 2
                    else:
                        matrix[i, j] = 1
                except:
                    pass

    total_tasks = len(task_names)
    total_models = len(models)
    total_valid_tasks = int(np.sum(np.all(matrix > 0, axis=1)))
    total_computed_cells = int(np.sum(matrix == 2))
    total_cached_cells = int(np.sum(matrix == 1))

    stats = {
        "total_tasks": total_tasks,
        "total_models": total_models,
        "total_valid_tasks": total_valid_tasks,
        "total_computed_cells": total_computed_cells,
        "total_cached_cells": total_cached_cells,
    }
    with open(thesis_file(output_file), "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved stats:", output_file)
    return stats


def main():
    print("Loading benchmark...")
    df_e_p = get_benchmark_dataframe(TARGET_DOMAINS, get_models)
    df_e_f = get_benchmark_dataframe(TARGET_DOMAINS, get_multilingual_models)
    df_m_p = get_benchmark_dataframe(TARGET_DOMAINS, get_models, True)
    df_m_f = get_benchmark_dataframe(TARGET_DOMAINS, get_multilingual_models, True)
    visualize_grid(df_e_p, "model_dataset_done.png")
    save_stats(df_e_p, "model_dataset_stats.json")
    visualize_grid(df_e_f, "model_dataset_done_full.png")
    save_stats(df_e_f, "model_dataset_stats_full.json")
    visualize_grid(df_m_f, "model_multilingual_dataset_done_full.png")
    save_stats(df_m_f, "model_multilingual_dataset_stats_full.json")
    visualize_grid(df_m_p, "model_multilingual_dataset_done.png")
    save_stats(df_m_p, "model_multilingual_dataset_stats.json")


if __name__ == "__main__":
    main()
