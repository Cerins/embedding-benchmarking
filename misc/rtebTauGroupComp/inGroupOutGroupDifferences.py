import pandas as pd
import re


def process_section(csv_file, section_name):
    df = pd.read_csv(csv_file)
    # Extract model name from markdown link if present
    df["Model"] = df["Model"].apply(
        lambda x: re.search(r"\[(.*?)\]", x).group(1)
        if isinstance(x, str) and re.search(r"\[(.*?)\]", x)
        else x
    )
    # Drop rows with any NaN values
    df = df.dropna(how="any")
    # Compute rank only if valid data remains
    df["Rank_" + section_name] = df["Mean (Task)"].rank(ascending=False, method="min")
    return df[["Model", "Rank_" + section_name]]


sections = {
    "Health": "health.csv",
    "Code": "code.csv",
    "Finance": "finance.csv",
    "Legal": "legal.csv",
}

ranks = []
for section, file in sections.items():
    ranks.append(process_section(file, section))

# Merge all ranks
merged = ranks[0]
for r in ranks[1:]:
    merged = pd.merge(merged, r, on="Model", how="outer")

# Average rank
rank_cols = [c for c in merged.columns if c.startswith("Rank_")]
merged["Average_Rank"] = merged[rank_cols].mean(axis=1)

# Remove any models with at least one NaN rank across sections
merged = merged.dropna(subset=rank_cols)

# Sort by average rank
merged = merged.sort_values(by="Average_Rank").reset_index(drop=True)

# print(merged.to_latex(index=False, float_format="%.2f"))

import pandas as pd
import numpy as np
import itertools
from scipy.stats import kendalltau
from tqdm import tqdm


# 1. Clean up and rank by metric metric
def process_detailed_file(path, group_name):
    df = pd.read_csv(path)
    metric_cols = df.select_dtypes(include="number").columns.drop(["Unnamed: 0"])
    df = df.dropna(subset=metric_cols).reset_index(drop=True)
    lists = []  # each list = one metric’s ranking vector
    for metric in metric_cols:
        # Sort descending (best = rank 1)
        temp = (
            df[["Model", metric]]
            .copy()
            .sort_values(by=metric, ascending=False)
            .reset_index(drop=True)
        )
        temp["Rank"] = temp.index + 1
        lists.append(
            {
                "group": group_name,
                "metric": metric,
                "ranking": dict(zip(temp["Model"], temp["Rank"])),
            }
        )
    return lists


# 2. Proocess all groups
domains = {
    "health": "health_detailed.csv",
    "code": "code_detailed.csv",
    "legal": "legal_detailed.csv",
    "finance": "finance_detailed.csv",
}

all_lists = []
for group, path in domains.items():
    all_lists.extend(process_detailed_file(path, group))

print(f"Total ranking lists: {len(all_lists)} ({', '.join(domains.keys())})")

# 3. Remove models, which do not posses a rank
all_models_sets = []
for l in all_lists:
    all_models_sets.append(set(l["ranking"].keys()))
common_models = set.intersection(*all_models_sets)
print(f"Models common across all metrics and domains: {len(common_models)}")

# Restrict rankings to only common models
for l in all_lists:
    # Restrict to common models
    filtered = {k: v for k, v in l["ranking"].items() if k in common_models}
    # Re-rank from 1..N based on the metric values or existing order
    # Since we already had descending order initially, we can sort by old rank
    sorted_models = sorted(filtered.items(), key=lambda x: x[1])
    l["ranking"] = {m: i + 1 for i, (m, _) in enumerate(sorted_models)}


# 4. Compute pairwise Kendall tau similarities
def kendall_distance(list_a, list_b):
    common = sorted(set(list_a.keys()) & set(list_b.keys()))
    if len(common) is not len(common_models):
        raise Exception("Hmmm")
    # 1 identicial, -1 sorted opposite
    ra = [list_a[m] for m in common]
    rb = [list_b[m] for m in common]
    tau, _ = kendalltau(ra, rb)
    return 1 - tau  # distance (1 - similarity)


# Creates all combinations of indexes, so for 0,1,2 it would be [0,1], [0,2], [1, 2] no duplicates, no same index reuse
pairs = list(itertools.combinations(range(len(all_lists)), 2))
results = []

for i, j in tqdm(pairs, desc="Computing pairwise distances"):
    l1, l2 = all_lists[i], all_lists[j]
    d = kendall_distance(l1["ranking"], l2["ranking"])
    if np.isnan(d):
        continue
    results.append(
        {
            "list1": l1["metric"],
            "list2": l2["metric"],
            "group1": l1["group"],
            "group2": l2["group"],
            "distance": d,
        }
    )

distances_df = pd.DataFrame(results)

# 5. Separate within-group vs between-group pairs
within = distances_df[distances_df["group1"] == distances_df["group2"]]["distance"]
between = distances_df[distances_df["group1"] != distances_df["group2"]]["distance"]

print(f"\nWithin-group pairs: {len(within)} | Between-group pairs: {len(between)}")
print(f"Mean distance (within):  {within.mean():.4f}")
print(f"Mean distance (between): {between.mean():.4f}")

observed_diff = between.mean() - within.mean()
print(f"\nObserved difference (between - within): {observed_diff:.4f}")

# --- Corrected permutation test ---
B = 4000
diffs = []

# Save the indices of list1/list2 for each distance row
distances_df["idx1"] = [pairs[k][0] for k in range(len(pairs))]
distances_df["idx2"] = [pairs[k][1] for k in range(len(pairs))]

groups = np.array([l["group"] for l in all_lists])

# print(groups, len(groups), all_lists)

for _ in tqdm(range(B), desc="Permutation test"):
    permuted = np.random.permutation(groups)
    perm_df = distances_df.copy()
    perm_df["group1_perm"] = permuted[perm_df["idx1"]].tolist()
    perm_df["group2_perm"] = permuted[perm_df["idx2"]].tolist()
    within_p = perm_df[perm_df["group1_perm"] == perm_df["group2_perm"]]["distance"]
    between_p = perm_df[perm_df["group1_perm"] != perm_df["group2_perm"]]["distance"]
    if len(within_p) > 0 and len(between_p) > 0:
        diffs.append(between_p.mean() - within_p.mean())

diffs = np.array(diffs)
p_value = (np.sum(diffs >= observed_diff) + 1) / (B + 1)

print(f"\nPermutation test p-value (within < between): {p_value:.4f}")
print("Small p-value -> rankings are more similar within-domain.")
