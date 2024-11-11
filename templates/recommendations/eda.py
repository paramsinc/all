import matplotlib.pyplot as plt
import numpy as np
import os

from config import config


def get_score_range_and_counts(interaction_data):
    """Prints the range of score values, count of items, and count of users.

    Args:
        interaction_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    all_scores = [e["score"] for e in interaction_data if "score" in e]
    if all_scores:
        min_score, max_score = min(all_scores), max(all_scores)
        print(f"Score Range: [{min_score}, {max_score}]")
    else:
        min_score, max_score = 1, 1
    return min_score, max_score


def display_score_histogram(interaction_data, save_fpath=None):
    """Displays a histogram of score values."""
    all_scores = [e["score"] for e in interaction_data if "score" in e]
    plt.hist(all_scores, bins="auto")
    plt.title("Histogram of Score Values")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    display_fig(save_fpath)


def display_score_count_per_item_histogram(interaction_data, save_fpath=None):
    """Displays a histogram of score counts per item."""
    item_counts = {}
    for event in interaction_data:
        item_id = event["item"]
        item_counts[item_id] = item_counts.get(item_id, 0) + 1
    plt.hist(item_counts.values(), bins="auto")
    plt.title("Histogram of Score Counts per Item")
    plt.xlabel("Score Count")
    plt.ylabel("Number of Items")
    display_fig(save_fpath)


def display_score_count_per_user_histogram(interaction_data, save_fpath=None):
    """Displays a histogram of score counts per user."""
    user_counts = {}
    for event in interaction_data:
        user_id = event["user"]
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
    plt.hist(user_counts, bins="auto")
    plt.title("Histogram of Score Counts per User")
    plt.xlabel("Score Count")
    plt.ylabel("Number of Users")
    display_fig(save_fpath)


def display_average_score_per_item_histogram(interaction_data, save_fpath=None):
    """Displays a histogram of average score per item."""
    item_scores = {}
    for event in interaction_data:
        item_id = event["item"]
        score = event.get("score", None)
        if score is not None:
            item_scores.setdefault(item_id, []).append(score)
    avg_scores = {item_id: np.mean(scores) for item_id, scores in item_scores.items()}
    plt.hist(avg_scores.values(), bins="auto")
    plt.title("Histogram of Average Score per Item")
    plt.xlabel("Average Score")
    plt.ylabel("Number of Items")
    display_fig(save_fpath)


def display_fig(save_fpath):
    if save_fpath:
        parent = os.path.dirname(save_fpath)
        os.makedirs(parent, exist_ok=True)
        plt.savefig(save_fpath, dpi=config.eda_figures_dpi)
    else:
        plt.show()


def display_all(interaction_data, user_data, item_data):
    fpath_base = config.eda_figures_dir
    min_score, max_score = get_score_range_and_counts(interaction_data)
    if min_score != max_score:
        display_score_histogram(interaction_data, save_fpath=f"{fpath_base}/scores.png")
        display_score_count_per_item_histogram(
            interaction_data, save_fpath=f"{fpath_base}/score_counts_per_item.png"
        )
        display_average_score_per_item_histogram(
            interaction_data, save_fpath=f"{fpath_base}/average_score_per_item.png"
        )
    # TODO: eda for user data, item data
