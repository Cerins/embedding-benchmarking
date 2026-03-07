import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sys
from collections import defaultdict
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
import mteb
from common.utils import *

mteb_tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])

task_map = {t.metadata.name: t for t in mteb_tasks}


def get_benchmark_dataframe(min_tasks=35, target_domains=TARGET_DOMAINS):
    cache = mteb.ResultCache(CACHE_DIR)
    # cache.download_from_remote()
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
        if good_task(t)
    ]
    tasks = [t for t in all_tasks if task_has_target_domain(t, target_domains)]
    task_to_domains = {task.metadata.name: get_task_domains(task) for task in tasks}
    task_names = set(task_to_domains.keys())
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    model_task_counts = defaultdict(int)
    model_scores = defaultdict(dict)
    print(os.path.realpath(results_dir))
    for model in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model)
        if not os.path.isdir(model_path):
            continue
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json") and file != "model_meta.json":
                    task = file[:-5]
                    if task not in task_names:
                        continue
                    path = os.path.join(root, file)
                    try:
                        with open(path) as f:
                            data = json.load(f)
                        if "scores" in data:
                            model_task_counts[model] += 1
                            model_scores[model][task] = True
                    except:
                        pass
    qualified_models = get_models()
    rows = []
    for task in sorted(task_names):
        row = {"task_name": task, "domains": task_to_domains[task]}
        non_nan_found = False
        for model in qualified_models:
            val = model_scores[model].get(task, np.nan)
            if not np.isnan(val):
                non_nan_found = True
            row[model] = val
        if non_nan_found:
            rows.append(row)
    df = pd.DataFrame(rows)
    columns = ["task_name", "domains"] + qualified_models
    return df[columns]


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
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("THESIS_PATH not set")
    os.chdir(thesis_path)
    plt.savefig(output_file, dpi=300)
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
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("THESIS_PATH not set")
    os.chdir(thesis_path)
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved stats:", output_file)
    return stats


def main():
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("THESIS_PATH not set")
    print("Loading benchmark...")
    df = get_benchmark_dataframe(30)
    visualize_grid(df, "model_dataset_done.png")
    stats = save_stats(df, "model_dataset_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
