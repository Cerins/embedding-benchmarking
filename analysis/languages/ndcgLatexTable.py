import pandas as pd

from common.utils import *
from analysis.languages.anova import build_long_df


# Build latex table
def build_latex_table(long, fn, label="tab:ndcg-task-model-multilingual"):
    ordered_tasks = (
        long.groupby("task_name")["score"].mean().sort_values().index.tolist()
    )

    model_order = (
        long.groupby("model_short")["score"].mean().sort_values().index.tolist()
    )

    pivot = long.groupby(["task_name", "model_short"])["score"].mean().unstack()
    pivot = pivot.loc[ordered_tasks, model_order]

    best_per_row = pivot.idxmax(axis=1)
    n_models = len(model_order)

    col_spec = "l" + "r" * n_models

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{nDCG@10 rezultāti daudzvalodu uzdevumiem. Treknrakstā --- labākais rezultāts uzdevumā.}"
    )
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \tiny")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")

    header_cells = ["Uzdevums"] + [
        rf"\rotatebox{{90}}{{\strut {latex_escape(model_short_name(name))}}}"
        for name in model_order
    ]
    lines.append("  " + " & ".join(header_cells) + r" \\")
    lines.append(r"  \midrule")

    for task_name in ordered_tasks:
        best_model = best_per_row[task_name]
        cells = [latex_escape(task_name)]
        for model in model_order:
            val = pivot.loc[task_name, model]
            if pd.isna(val):
                cells.append("--")
            elif model == best_model:
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


def run(gm, suffix):
    df = get_multilingual_scores_dataframe(gm)
    long = build_long_df(df)
    build_latex_table(
        long, thesis_file(f"ndcg_task_model_multilingual_table{suffix}.tex")
    )


def main():
    run(get_multilingual_models, "")


if __name__ == "__main__":
    main()
