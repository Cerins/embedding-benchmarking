import mteb
from collections import Counter
import matplotlib.pyplot as plt


def good_task(t):
    if t.metadata.category != "t2t":
        return False
    return True


def suggested_metrics_histogram(tasks):
    all_metrics = []
    for task in tasks:
        if hasattr(task.metadata, "main_score"):
            metrics = task.metadata.main_score
            all_metrics.append(task.metadata.main_score)
        else:
            raise Exception("Not possible")

    counts = Counter(all_metrics)
    total = sum(counts.values())
    labels, values = zip(*counts.most_common())
    percentages = [(v / total) * 100 for v in values]
    # Plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Metrika")
    plt.ylabel("Skaits")
    plt.title("MTEB datukopu ieteiktās metrikas")
    plt.grid(axis="y", alpha=0.3)
    # Add labels: count + percentage
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f"{values[i]}\n({percentages[i]:.1f}%)"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig("metrics_histogram.png", dpi=300)
    print("\nSaved metric histogram")
    plt.close()


def main():
    tasks = [t for t in mteb.get_tasks(task_types=["Retrieval"]) if good_task(t)]
    suggested_metrics_histogram(tasks)


if __name__ == "__main__":
    main()
