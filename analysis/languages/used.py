import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
from common.utils import *

# Consolidate dialect/script variants into one canonical code
LANG_GROUPS = {
    # Arabic dialects
    "acm": "ara",
    "apc": "ara",
    "arb": "ara",
    "ars": "ara",
    "ary": "ara",
    "arz": "ara",
    # Chinese
    "cmn": "zho",
    # Norwegian
    "nob": "nor",
    "nno": "nor",
    # Malay
    "zsm": "msa",
    # Persian
    "pes": "fas",
    # Uzbek
    "uzn": "uzb",
    # Swahili
    "swh": "swa",
}

# Languages appearing in fewer than this many tasks are merged into "other"
MIN_TASK_COUNT = 4


# Normalize codes
def normalize_languages(languages):
    return list({LANG_GROUPS.get(l, l) for l in languages})


def visualize_language_overlaps(df, fn):
    # Apply grouping to every task's language list
    df = df.copy()
    df["languages"] = df["languages"].apply(normalize_languages)

    # Count how many tasks each language appears in
    lang_counts = {}
    for _, row in df.iterrows():
        for lang in row.languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    frequent = {l for l, c in lang_counts.items() if c >= MIN_TASK_COUNT}

    # Replace rare languages with "other" (deduplicated per task)
    def remap(langs):
        result = set()
        for l in langs:
            result.add(l if l in frequent else "other")
        return list(result)

    df["languages"] = df["languages"].apply(remap)

    target_languages = sorted(frequent) + ["other"]
    # Drop "other" column if no task actually has rare-only languages
    if not any("other" in row.languages for _, row in df.iterrows()):
        target_languages = sorted(frequent)

    # Put membership data for upset
    membership = pd.DataFrame(False, index=df.index, columns=target_languages)
    for idx, row in df.iterrows():
        for lang in row.languages:
            if lang in membership.columns:
                membership.loc[idx, lang] = True

    upset_data = from_indicators(target_languages, membership)

    # Plot upset
    plt.figure(figsize=(max(12, len(target_languages) * 1.2), 6))
    up = UpSet(
        upset_data,
        subset_size="count",
        sort_by="cardinality",
        show_counts="{:d}",
        facecolor="C0",
        other_dots_color=0.3,
        shading_color=0.1,
        element_size=28,
    )
    subplots = up.plot()
    subplots["intersections"].set_ylabel("Uzdevumu skaits")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to '{fn}'")


def main():
    print("Loading multilingual benchmark data...")
    df = get_multilingual_scores_dataframe(get_multilingual_models)
    print(df[["task_name", "languages"]])

    visualize_language_overlaps(df, thesis_file("lang_task_language_overlap.png"))


if __name__ == "__main__":
    main()
