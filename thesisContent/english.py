import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from itertools import combinations
from collections import defaultdict
from sklearn.manifold import MDS
import mteb
import os
import json
import sys
from upsetplot import UpSet, from_indicators


CACHE_DIR = "./cache"

TARGET_DOMAINS_UNUSED = [
    "legal",
    "medical",
    "chemistry",
    "engineering",
    "programming",
    "financial",
    "fiction",
]

TARGET_DOMAINS = [
    "legal",
    "medical",
    # "chemistry",  # These 2 have large overlap with legal
    # "engineering",
    "programming",
    "financial",
    # "fiction", # Not enoguh items to make sense to use
]


def good_task(t):
    # Only allow t2t
    if t.metadata.category != "t2t":
        return False
    # Find if there exists language that is 3 letters long and not eng
    for lang in t.metadata.languages:
        if len(lang) == 3 and lang != "eng":
            return False
    return True


def get_task_domains(task):
    if not hasattr(task.metadata, "domains") or not task.metadata.domains:
        return []
    domains = []
    for domain in task.metadata.domains:
        dl = domain.lower()
        # Basically only show the interesting domains
        # if dl in TARGET_DOMAINS_UNUSED:
        domains.append(dl)
    return domains


def task_has_target_domain(task, target_domains):
    domains = get_task_domains(task)
    for td in target_domains:
        print(td, target_domains, domains)
        if td in domains:
            return True
    return False


def get_benchmark_dataframe(min_tasks=35, target_domains=TARGET_DOMAINS):
    cache = mteb.ResultCache(CACHE_DIR)
    cache.download_from_remote()
    # Load and filter tasks
    all_tasks = [
        t
        for t in mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
        if good_task(t)
    ]
    # Filter to only tasks with TARGET_DOMAINS
    tasks = [t for t in all_tasks if task_has_target_domain(t, target_domains)]
    print(f"Total tasks with target domains: {len(tasks)}")
    print(f"Target domains: {target_domains}")
    # Create task name to domain mapping
    task_to_domains = {task.metadata.name: get_task_domains(task) for task in tasks}
    task_names = set(task_to_domains.keys())
    print("Tasks by domain:")
    for domain in target_domains:
        count = sum(1 for d in task_to_domains.values() if domain in d)
        print(f"  {domain}: {count} tasks")
    # Scan models and count tasks
    results_dir = os.path.join(CACHE_DIR, "remote", "results")
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None
    print("Scanning models...")
    # Get all model directories
    model_dirs = []
    for item in os.listdir(results_dir):
        model_path = os.path.join(results_dir, item)
        if os.path.isdir(model_path):
            model_dirs.append((item, model_path))
    # Count tasks per model and collect ndcg_at_10 scores
    model_task_counts = defaultdict(int)
    model_scores = defaultdict(dict)  # model -> {task_name: ndcg_at_10}
    for model_name, model_path in model_dirs:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json") and file != "model_meta.json":
                    task_name = file[:-5]  # Remove .json extension
                    # Check if this task is in our filtered task list
                    if task_name in task_names:
                        json_path = os.path.join(root, file)
                        try:
                            with open(json_path, "r") as f:
                                data = json.load(f)
                            if (
                                "scores" in data
                                and "test" in data["scores"]
                                and len(data["scores"]["test"]) > 0
                            ):
                                test_scores = data["scores"]["test"][0]
                                if "ndcg_at_10" in test_scores:
                                    model_task_counts[model_name] += 1
                                    model_scores[model_name][task_name] = test_scores[
                                        "ndcg_at_10"
                                    ]
                        except (json.JSONDecodeError, KeyError, IOError):
                            continue
    # Filter models with at least min_tasks
    qualified_models = [
        model for model, count in model_task_counts.items() if count >= min_tasks
    ]
    print(f"Total models found: {len(model_dirs)}")
    print(f"Models with >= {min_tasks} tasks: {len(qualified_models)}")
    if not qualified_models:
        print(f"No models with >= {min_tasks} tasks found!")
        return None
    print("Qualified models:")
    for model in sorted(qualified_models):
        print(f"  {model}: {model_task_counts[model]} tasks")
    # Create the results matrix
    rows = []
    for task_name in sorted(task_names):
        domains = task_to_domains[task_name]
        row = {"task_name": task_name, "domains": domains}
        # Add scores for each qualified model
        for model in sorted(qualified_models):
            score = model_scores[model].get(task_name, np.nan)
            row[model] = score
        rows.append(row)
    # Create DataFrame
    df = pd.DataFrame(rows)
    # Reorder columns: task_name, domain, then models
    columns = ["task_name", "domains"] + sorted(qualified_models)
    df = df[columns]
    print(f"Initial matrix shape: {df.shape}")
    print(f"Tasks before filtering: {len(df)}")
    print(f"Models: {len(qualified_models)}")
    # Filter to only tasks where all models have scores (no missing values)
    df_complete = df.dropna()
    df_complete = df
    print("After filtering for complete data:")
    print(f"Tasks with all model scores: {len(df_complete)}")
    print(f"Removed tasks: {len(df) - len(df_complete)}")
    return df_complete


def domain_to_index(domain, target_domains):
    for i, d in enumerate(target_domains):
        if d == domain:
            return i
    raise Exception(f"Bad index ${domain} ${target_domains}")


def vizualize_domain_overlaps(df, target_domains, fn):
    if target_domains is None:
        # Special processing get literally all domains
        target_domains = []
        for idx, row in df.iterrows():
            for domain in row.domains:
                if domain not in target_domains:
                    target_domains.append(domain)
    membership = pd.DataFrame(False, index=df.index, columns=target_domains)

    for idx, row in df.iterrows():
        for domain in row.domains:
            if domain in membership.columns:
                membership.loc[idx, domain] = True

    # Convert to proper UpSet input
    upset_data = from_indicators(target_domains, membership)

    # Plot
    plt.figure(figsize=(12, 6))
    up = UpSet(
        upset_data,
        subset_size="count",
        sort_by="cardinality",
        show_counts="{:d}",  # formatted counts
        facecolor="C0",  # standard matplotlib blue
        other_dots_color=0.3,
        shading_color=0.1,
        element_size=28,
    )
    subplots = up.plot()
    subplots["intersections"].set_ylabel("Pārklājuma izmērs")
    plt.suptitle("Domēnu pārklājums (UpSet diagramma)")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to '{fn}'")


def main():
    thesis_path = os.environ.get("THESIS_PATH")
    if not thesis_path:
        sys.exit("Error: THESIS_PATH environment variable is not set.")
    print("Loading benchmark data...")
    df = get_benchmark_dataframe(35)
    pd.set_option("display.max_columns", None)
    # print(df.head())
    os.chdir(thesis_path)
    vizualize_domain_overlaps(df, TARGET_DOMAINS, "chosen_domains_overlap.png")
    vizualize_domain_overlaps(df, TARGET_DOMAINS_UNUSED, "full_domains_overlap.png")
    vizualize_domain_overlaps(df, None, "unclean_domains_overlap.png")
    if df is None or df.empty:
        print("Failed to load benchmark data")
        return


if __name__ == "__main__":
    main()
