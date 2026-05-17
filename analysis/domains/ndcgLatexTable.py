import pandas as pd

from common.utils import *
from analysis.domains.anova import build_long_df


# Build the latex table
def build_latex_table(long, fn, label="tab:ndcg-task-model"):
    task_domain = (
        long[["task_name", "domain"]]
        .drop_duplicates()
        .sort_values(["domain", "task_name"])
    )
    ordered_tasks = task_domain["task_name"].tolist()
    task_labels = [f"{r.task_name} ({r.domain})" for r in task_domain.itertuples()]
    model_order = (
        long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    )
    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    pivot = pivot.loc[ordered_tasks, model_order]
    pivot.index = task_labels
    best_per_row = pivot.idxmax(axis=1)
    n_models = len(model_order)
    col_spec = "l" + "r" * n_models
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{nDCG@10 rezultāti angļu valodas uzdevumiem. Treknrakstā --- labākais rezultāts uzdevumā.}"
    )
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")
    header_cells = ["Uzdevums"] + [
        rf"\rotatebox{{90}}{{\strut {latex_escape(model_short_name(name))}}}"
        for name in model_order
    ]
    lines.append("  " + " & ".join(header_cells) + r" \\")
    lines.append(r"  \midrule")
    prev_domain = None
    for task_label, task_name in zip(task_labels, ordered_tasks):
        domain = task_domain.loc[task_domain["task_name"] == task_name, "domain"].iloc[
            0
        ]
        if prev_domain is not None and domain != prev_domain:
            lines.append(r"  \midrule")
        prev_domain = domain
        best_model = best_per_row[task_label]
        cells = [latex_escape(task_label)]
        for model in model_order:
            val = pivot.loc[task_label, model]
            if pd.isna(val):
                cells.append("--")
            elif model == best_model:
                cells.append(rf"\textbf{{{val:.3f}}}")
            else:
                cells.append(f"{val:.3f}")
        lines.append("  " + " & ".join(cells) + r" \\")
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    content = "\n".join(lines) + "\n"
    with open(fn, "w") as f:
        f.write(content)
    print(f"LaTeX table saved to '{fn}'")
    print(content)


def run(gm, suffix):
    df = get_scores_dataframe(gm)
    long = build_long_df(df)
    build_latex_table(long, thesis_file(f"ndcg_task_model_table{suffix}.tex"))


def main():
    run(get_models, "")


if __name__ == "__main__":
    main()
