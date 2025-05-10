import os
import pandas as pd
import matplotlib.pyplot as plt
from plot import process_results


# compute proportion optimal action chosen for each arm + hint
def compute_proportions(config, results):
    df = process_results(config, results)
    df["optimal_prop"] = df.groupby("og_idx")["optimal_a"].cumsum() / (df["t"] + 1)
    df["og_arms"] = df["og_arms"].apply(str)
    return df


# rank the hints by how much they help the model
def compute_ranks(df):
    # Group by og_arms and og_hints, and compute the mean optimal_prop for each group.
    grouped = df.groupby(["og_arms", "og_hints"])["optimal_prop"].mean().reset_index()

    # Extract the baseline for each og_arms (where og_hints == "no hint")
    baseline = grouped[grouped["og_hints"] == "no hint"][
        ["og_arms", "optimal_prop"]
    ].rename(columns={"optimal_prop": "baseline_optimal_prop"})

    # Merge the baseline values back onto the grouped dataframe based on og_arms.
    comparison = pd.merge(grouped, baseline, on="og_arms", how="left")

    # Compute the difference relative to the "no hint" condition.
    comparison["difference"] = (
        comparison["optimal_prop"] - comparison["baseline_optimal_prop"]
    )

    # Rank the hints by the "difference" column for each "og_arms" configuration.
    comparison["rank"] = comparison.groupby("og_arms")["difference"].rank(
        method="dense", ascending=False
    )

    # Sort the results for better readability.
    ranked_comparison = comparison.sort_values(["og_arms", "rank"])

    # Display the ranked comparison table.
    print(ranked_comparison)
    return ranked_comparison


def plot_rankings(ranked_comparison, type):
    # Get the unique og_arms values.
    og_arms_values = ranked_comparison["og_arms"].unique()

    # Set up the figure: one subplot per og_arms configuration.
    n = len(og_arms_values)
    fig, axs = plt.subplots(n, 1, figsize=(8, 4 * n), constrained_layout=True)
    fig.suptitle(f"Hint Rankings for {type} Bandit")
    if n == 1:
        axs = [axs]  # ensure axs is iterable when only one subplot

    for ax, og_arm in zip(axs, og_arms_values):
        # Filter data for the current og_arms configuration.
        data = ranked_comparison[ranked_comparison["og_arms"] == og_arm]

        # Sort data by rank (or difference) so that the best hint is at the top.
        data = data.sort_values("rank")

        # Create a horizontal bar plot.
        ax.barh(data["og_hints"], data["difference"], color="skyblue")
        ax.set_xlabel("Diff in Optimal Actions")
        ax.set_title(f"Arms: {og_arm}")

        # Invert the y-axis to have the highest rank at the top.
        ax.invert_yaxis()

    plt.savefig(f"results/{type}_rankings.png")


def get_most_recent_file(partial_str, directory="."):
    """
    Searches for files in the given directory whose names contain partial_str.
    Returns the name of the most recently modified file among them.

    Parameters:
    - partial_str: The substring to look for in the file names.
    - directory: The directory in which to search (defaults to the current directory).

    Returns:
    - The file name of the most recently modified matching file, or None if no match is found.
    """
    # List all items in the directory
    items = os.listdir(directory)

    # Filter items: must be a file and contain the partial string in its name
    matching_files = [
        item
        for item in items
        if partial_str in item and os.path.isfile(os.path.join(directory, item))
    ]

    if not matching_files:
        return None

    # Get the file with the latest modification time
    most_recent = max(
        matching_files, key=lambda f: os.path.getmtime(os.path.join(directory, f))
    )

    print(most_recent)
    return most_recent


def main():
    for type in ["stationary", "drifting", "stepwise", "moving_avg", "time_delayed"]:
        config = f"config/{type}.json"
        result = get_most_recent_file(f"{type}_2025", "results")
        if result:
            df = compute_proportions(config, f"results/{result}")
            rankings = compute_ranks(df)
            plot_rankings(rankings, type)
        else:
            print(f"No results found for {type}.")


if __name__ == "__main__":
    main()
