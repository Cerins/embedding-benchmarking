# Output a pie chart of suggested metrics
import mteb
from collections import Counter
import matplotlib.pyplot as plt
from common.utils import thesis_file


# Only filter for main metric ndcg
def good_task(t):
    return t.metadata.category == "t2t"


def suggested_metrics_histogram(tasks):
    # Collect main_score metrics
    metrics = [
        t.metadata.main_score for t in tasks if hasattr(t.metadata, "main_score")
    ]
    # Array to name -> count metric
    counts = Counter(metrics)
    # Separate metrics with count > 1 and <= 1
    # To basically make more readable pie chart
    filtered_counts = {}
    other_count = 0
    # Count
    for metric, count in counts.items():
        if count <= 1:
            other_count += count
        else:
            filtered_counts[metric] = count
    if other_count > 0:
        filtered_counts["Cits"] = other_count
    # Get total count
    total = sum(filtered_counts.values())
    # Sort by frequency (descending)
    sorted_data = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    percentages = [(v / total) * 100 for v in values]
    # Create legend labels
    legend_labels = [
        f"{label}: {value} ({pct:.1f}%)"
        for label, value, pct in zip(labels, values, percentages)
    ]
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    # Draw pie
    wedges, _ = ax.pie(values, startangle=140)
    ax.legend(
        wedges,
        legend_labels,
        title="Metrikas",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.tight_layout()
    plt.savefig(thesis_file("metrics_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved metric histogram")


def main():
    # Retrieve tasks and run them
    tasks = [t for t in mteb.get_tasks(task_types=["Retrieval"]) if good_task(t)]
    suggested_metrics_histogram(tasks)


if __name__ == "__main__":
    main()
