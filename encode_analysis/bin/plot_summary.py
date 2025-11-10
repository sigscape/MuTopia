#!/usr/bin/env python3

# for now just plot a random matplotlib figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mutopia.tuning import load_study
import mutopia.analysis as mu

def plot_summary(study_dir: str, output_file: str) -> None:

    study, *_ = load_study(study_dir)

    best_values = pd.Series(
        {
            trial.number : max(trial.intermediate_values.values(), default=float("nan"))
            for trial in study.trials
        }
    )
    results = study.trials_dataframe().join(best_values.rename("best_value"), how="left")
    results["best_value"] = np.where(~np.isfinite(results["best_value"]), results["value"], results["best_value"])
    results = results.dropna(subset=["best_value"])

    print("Summary of results:")
    print(results[["number", "best_value"]].head(4))

    _, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(
        data=results,
        x="params_num_components",
        y="best_value",
        style="state",
        hue="state",
        palette=mu.pl.categorical_palette,
        style_order=["COMPLETE", "PRUNED", "RUNNING", "FAIL"],
        ax=ax,
    )
    
    top = results["best_value"].max()*1.001
    bottom = results[results["state"].isin(["COMPLETE","RUNNING"])]["best_value"].min()*0.999
    ax.set_ylim(
        bottom=bottom,
        top=top,
    )
    # Label the maximum value point for each params_num_components
    for num_components in results.dropna(subset=["best_value"])["params_num_components"].unique():
        group = results[(results["params_num_components"] == num_components) & (results["best_value"] >= bottom)]
        if len(group) > 0:
            max_idx = group["best_value"].idxmax()
            max_row = group.loc[max_idx]
            ax.text(
                max_row["params_num_components"],
                max_row["best_value"],
                f' {max_row["number"]}',
                verticalalignment='bottom',
                horizontalalignment='left',
                fontsize=8,
                alpha=0.7
            )
    
    ax.legend(
        title="Trial status",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    ax.set(
        xlabel="Number of components",
        ylabel="Test score \u2192",
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot summary of tuning study results."
    )
    parser.add_argument(
        "study_dir",
        type=str,
        help="Path to the tuning study directory."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output plot."
    )
    args = parser.parse_args()
    plot_summary(args.study_dir, args.output_file)