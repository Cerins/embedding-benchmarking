# Script which vizualizes which domains are chosen for the thesis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
from common.utils import *


# Map domain to index to use for upsetplot
def domain_to_index(domain, target_domains):
    for i, d in enumerate(target_domains):
        if d == domain:
            return i
    raise Exception(f"Bad index ${domain} ${target_domains}")


# Vizualzie domain overlap
def vizualize_domain_overlaps(df, target_domains, fn):
    if target_domains is None:
        # Special processing get literally all domains
        target_domains = []
        for idx, row in df.iterrows():
            for domain in row.domains:
                if domain not in target_domains:
                    target_domains.append(domain)
    # State membership
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
    # plt.suptitle("Domēnu pārklājums (UpSet diagramma)")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to '{fn}'")


def main():
    print("Loading benchmark data...")
    df = get_scores_dataframe().dropna()
    pd.set_option("display.max_columns", None)
    print(df)
    vizualize_domain_overlaps(
        df, TARGET_DOMAINS, thesis_file("chosen_domains_overlap.png")
    )
    vizualize_domain_overlaps(
        df, TARGET_DOMAINS_UNUSED, thesis_file("full_domains_overlap.png")
    )
    vizualize_domain_overlaps(df, None, thesis_file("unclean_domains_overlap.png"))


if __name__ == "__main__":
    main()
