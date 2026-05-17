import pandas as pd

from common.utils import (
    thesis_file,
    mteb_to_hf_repo,
    model_short_name,
    latex_escape,
    get_latvian_models,
    get_ailab_models,
)
from analysis.lv.compare import TASKS, load_scores


def build_latex_table(
    model_scores, task_names, fn, label="tab:ndcg-task-model-latvian"
):
    # rows = models (descending mean score), cols = tasks (alphabetical)
    ordered_tasks = sorted(task_names)

    model_means = {
        mteb_to_hf_repo(m): sum(scores[t] for t in task_names) / len(task_names)
        for m, scores in model_scores.items()
    }
    model_order = sorted(model_means, key=model_means.get, reverse=True)

    # mteb-format model name -> hf-format for lookup
    mteb_by_hf = {mteb_to_hf_repo(m): m for m in model_scores}

    pivot = pd.DataFrame(
        {
            hf: {t: model_scores[mteb_by_hf[hf]][t] for t in ordered_tasks}
            for hf in model_order
        }
    ).T  # rows = models, cols = tasks
    pivot = pivot.loc[model_order, ordered_tasks]

    best_per_col = pivot.idxmax(axis=0)
    n_tasks = len(ordered_tasks)

    col_spec = "l" + "r" * n_tasks

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{nDCG@10 rezultāti latviešu valodas uzdevumiem. Treknrakstā — labākais rezultāts uzdevumā.}"
    )
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \small")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")

    header_cells = ["Modelis"] + [latex_escape(name) for name in ordered_tasks]
    lines.append("  " + " & ".join(header_cells) + r" \\")
    lines.append(r"  \midrule")

    for model in model_order:
        cells = [latex_escape(model_short_name(model))]
        for task_name in ordered_tasks:
            val = pivot.loc[model, task_name]
            if pd.isna(val):
                cells.append("--")
            elif model == best_per_col[task_name]:
                cells.append(rf"\textbf{{{val:.3f}}}")
            else:
                cells.append(f"{val:.3f}")
        lines.append("  " + " & ".join(cells) + r" \\")

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    with open(fn, "w") as f:
        f.write(content)
    print(f"LaTeX table saved to '{fn}'")
    print(content)


def main():
    allowed = set(get_latvian_models() + get_ailab_models())
    model_scores = {m: s for m, s in load_scores(TASKS).items() if m in allowed}
    print(f"Models with full coverage: {len(model_scores)}")
    build_latex_table(
        model_scores, TASKS, thesis_file("ndcg_task_model_latvian_table.tex")
    )


if __name__ == "__main__":
    main()
