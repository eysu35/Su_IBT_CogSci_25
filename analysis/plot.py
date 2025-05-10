import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import json
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import ast


# adapt centaur results to be same as model results
def process_results(config_path, result_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    df = pd.read_parquet(result_path)
    # labels = ["neutral"] * config["ntrials"] + ["hint"] * 11 * config["ntrials"]
    labels = (
        ["neutral"] * config["ntrials"]
        + ["hint"] * 5 * config["ntrials"]
        + ["strategy"] * 5 * config["ntrials"]
    )

    labels = labels * config["narms"] * 3

    df["labels"] = labels

    df["str_arms"] = df["arms"].apply(str)
    # df['arms'] = df['arms'].apply(lambda x: np.array(ast.literal_eval(x)))
    df["original_arm_values"] = df["og_arms"].apply(str)
    df["og_idx"] = df.index
    df = df.explode("history").reset_index(drop=True)
    df["t"] = df.groupby("og_idx").cumcount()
    df.rename(columns={"history": "a", "hint": "agent"}, inplace=True)
    df["optimal"] = df["arms"].apply(lambda x: np.argmax(x) + 1)
    df["optimal_a"] = df["a"] == df["optimal"]

    return df


def plot(config, df):
    mpl.rcParams.update(
        {
            "font.size": 18,  # Controls default text size
            "axes.titlesize": 18,  # Title font size
            "axes.labelsize": 16,  # X and Y labels font size
            "xtick.labelsize": 16,  # X tick labels font size
            "ytick.labelsize": 16,  # Y tick labels font size
            "legend.fontsize": 20,  # Legend font size
        }
    )

    unique_arms = df["original_arm_values"].unique()

    fig, axes = plt.subplots(3, 1, figsize=(8, 6 * 3), dpi=300)

    # unique_arms = ['[ 4  8 16 32 64]', '[10 20 40 60 70]', '[10 30 32 65 70]' ]
    # unique_arms_title = ['[4 8 16 32 64]', '[10 20 40 60 70]', '[10 30 32 65 70]' ]

    for i, ax in enumerate(axes):
        temp = df[df["original_arm_values"] == unique_arms[i]]
        sns.lineplot(
            ax=ax,
            data=temp[temp["agent"] == "no hint"],
            x="t",
            y="optimal_a",
            # color='grey',
            linestyle="--",
            linewidth=3,
            hue="og_hints",
            palette=["grey"],
            errorbar=None,
        )

        sns.lineplot(
            ax=ax,
            data=temp[temp["labels"] == "hint"],
            x="t",
            y="optimal_a",
            hue="og_hints",
            # hue_order=hints,
            palette="Greens_r",
            errorbar=None,
        )
        sns.lineplot(
            ax=ax,
            data=temp[temp["labels"] == "strategy"],
            x="t",
            y="optimal_a",
            hue="og_hints",
            palette="Blues_r",
            errorbar=None,
        )
        ax.set_title(f"Original Arm Values: {unique_arms[i]}")
        ax.set_xticks(range(0, 20))
        ax.set_xticklabels(range(0, 20))
        if i == 2:
            ax.set_xlabel("Trial")
        else:
            ax.set_xlabel("")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        ax.set_ylabel(
            "Proportion\noptimal\naction\nchosen", labelpad=30, rotation=0, va="center"
        )
        legend = ax.legend()
        for t in legend.get_texts():
            t.set_text(
                t.get_text()
                .replace("Hint: ", "")
                .replace("arm 4", "arm 5")
                .replace("arm 3", "arm 4")
                .replace("arm 2", "arm 3")
                .replace("arm 1", "arm 2")
                .replace("arm 0", "arm 1")
                .replace("&nbsp;", "")
                .capitalize()
            )
        legend.set_title("Hint/Neutral/Lie")
        # move legend to underneath the last plot
        if i == 2:
            sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
        else:
            # no legend
            ax.legend().set_visible(False)

        # ax.get_legend().remove()
        plt.savefig("results/" + config["bandit"] + ".png", bbox_inches="tight")
        # plt.show()


def plot_results(config, results):
    df = process_results(config, results)
    plot(config, df)


if __name__ == "__main__":
    plot_results("config/stationary.json", "results/stationary_20250323-011007.parquet")
    plot_results("config/drifting.json", "results/drifting_20250323-073043.parquet")
    plot_results("config/stepwise.json", "results/stepwise_20250323-103104.parquet")
    plot_results("config/moving_avg.json", "results/moving_avg_20250323-133114.parquet")
    plot_results(
        "config/time_delayed.json", "results/time_delayed_20250323-163021.parquet"
    )
