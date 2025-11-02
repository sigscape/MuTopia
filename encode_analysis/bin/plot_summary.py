#!/usr/bin/env python3

# for now just plot a random matplotlib figure
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mutopia.tuning import load_study
import mutopia.analysis as mu

def plot_summary(study_dir: str, output_file: str) -> None:

    study, *_ = load_study(study_dir)
    results = study.trials_dataframe()

    _, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(
        data=results,
        x="params_num_components",
        y="value",
        style="state",
        hue="state",
        palette=mu.pl.categorical_palette,
    )
    
    # Label the maximum value point for each params_num_components
    for num_components in results.dropna(subset=["value"])["params_num_components"].unique():
        group = results[results["params_num_components"] == num_components]
        if len(group) > 0:
            max_idx = group["value"].idxmax()
            max_row = group.loc[max_idx]
            ax.text(
                max_row["params_num_components"],
                max_row["value"],
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