import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
import json
from common.utils import *

from analysis.domains.anova import build_long_df


# Basically need to get model sizes, model, its avg score and log of param size
def build_param_df(long):
    avg = long.groupby("model")["score"].mean().reset_index()
    avg.columns = ["model", "avg_score"]
    avg["param_size"] = avg["model"].map(
        lambda m: MODEL_TO_PARAM_SIZE.get(hf_repo_to_mteb(m))
    )
    avg["model_short"] = avg["model"]
    # Drop models without known param size
    known = avg.dropna(subset=["param_size"]).copy()
    known["log_params"] = np.log10(known["param_size"])
    return known


# Do a linear model
def fit_correlation(df):
    model = smf.ols("avg_score ~ log_params", data=df).fit()
    return model


# Return outliers via IQR on OLS
def detect_outliers(df, fit):
    resid = fit.resid
    q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
    iqr = q3 - q1
    return (resid < q1 - 1.5 * iqr) | (resid > q3 + 1.5 * iqr)


def plot_correlation(df_in, df_out, fit, fn):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_in["log_params"], df_in["avg_score"], zorder=3, label="Modeļi")
    if len(df_out) > 0:
        ax.scatter(
            df_out["log_params"],
            df_out["avg_score"],
            color="red",
            marker="x",
            s=100,
            linewidths=2,
            zorder=4,
            label="Izslēgtie modeļi (izlecēji)",
        )
        for _, row in df_out.iterrows():
            ax.annotate(
                f"{row['model_short']}\n(izslēgts)",
                (row["log_params"], row["avg_score"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color="red",
            )

    x_range = np.linspace(df_in["log_params"].min(), df_in["log_params"].max(), 200)
    y_pred = fit.params["Intercept"] + fit.params["log_params"] * x_range
    ax.plot(
        x_range, y_pred, color="C1", label=f"Lineāra regresija (R²={fit.rsquared:.3f})"
    )

    for _, row in df_in.iterrows():
        ax.annotate(
            row["model_short"],
            (row["log_params"], row["avg_score"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )

    ticks = [1e7, 1e8, 1e9, 1e10]
    tick_labels = ["10M", "100M", "1B", "10B"]
    ax.set_xticks([np.log10(t) for t in ticks])
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Parametru skaits")
    ax.set_ylabel("Vidējais nDCG@10")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Param correlation plot saved to '{fn}'")


def save_stats(df_in, df_out, fit, fn):
    stats = {
        "r_squared": fit.rsquared,
        "p_value_log_params": float(fit.pvalues["log_params"]),
        "coef_log_params": float(fit.params["log_params"]),
        "n_models": len(df_in),
        "n_outliers_removed": len(df_out),
        "removed_models": df_out["model_short"].tolist(),
    }
    with open(fn, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to '{fn}'")


def main():
    df = get_scores_dataframe(get_multilingual_models).dropna()
    long = build_long_df(df)
    param_df = build_param_df(long)
    print(f"Models with known param size: {len(param_df)}")
    print(param_df[["model_short", "param_size", "avg_score"]].to_string(index=False))

    # All models
    fit_all = fit_correlation(param_df)
    empty = param_df.iloc[0:0]  # empty df with same columns
    plot_correlation(param_df, empty, fit_all, thesis_file("param_correlation.png"))
    save_stats(param_df, empty, fit_all, thesis_file("param_correlation_stats.json"))

    # Outliers removed
    outlier_mask = detect_outliers(param_df, fit_all)
    param_inliers = param_df[~outlier_mask].copy()
    param_outliers = param_df[outlier_mask].copy()
    print(
        f"Outliers removed ({len(param_outliers)}): {param_outliers['model_short'].tolist()}"
    )

    fit_clean = fit_correlation(param_inliers)
    print(fit_clean.summary())
    plot_correlation(
        param_inliers,
        param_outliers,
        fit_clean,
        thesis_file("param_correlation_no_outliers.png"),
    )
    save_stats(
        param_inliers,
        param_outliers,
        fit_clean,
        thesis_file("param_correlation_no_outliers_stats.json"),
    )


if __name__ == "__main__":
    main()
