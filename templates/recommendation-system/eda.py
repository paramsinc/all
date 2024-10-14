import matplotlib.pyplot as plt
import numpy as np
import os

import data
from config import config


def get_score_range_and_counts(score_data):
    """Prints the range of score values, count of items, and count of users.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    all_scores = [score for scores in score_data.values() for _, score in scores]
    min_score, max_score = min(all_scores), max(all_scores)
    num_items = len(
        set(item_id for scores in score_data.values() for item_id, _ in scores)
    )
    num_users = len(score_data)
    print(f"Score Range: [{min_score}, {max_score}]")
    print(f"Number of Items: {num_items}")
    print(f"Number of Users: {num_users}")


def display_score_histogram(score_data, suffix, save_fpath=None):
    """Displays a histogram of score values.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    all_scores = [score for scores in score_data.values() for _, score in scores]
    plt.hist(all_scores, bins="auto")
    plt.title("Histogram of Score Values" + suffix)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    display_fig(save_fpath)


def display_score_count_per_item_histogram(score_data, suffix, save_fpath=None):
    """Displays a histogram of score counts per item.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    item_counts = {}
    for scores in score_data.values():
        for item_id, _ in scores:
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
    plt.hist(item_counts.values(), bins="auto")
    plt.title("Histogram of Score Counts per Item" + suffix)
    plt.xlabel("Score Count")
    plt.ylabel("Number of Items")
    display_fig(save_fpath)


def display_score_count_per_user_histogram(score_data, suffix, save_fpath=None):
    """Displays a histogram of score counts per user.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    user_counts = [len(scores) for scores in score_data.values()]
    plt.hist(user_counts, bins="auto")
    plt.title("Histogram of Score Counts per User" + suffix)
    plt.xlabel("Score Count")
    plt.ylabel("Number of Users")
    display_fig(save_fpath)


def display_average_score_per_item_histogram(score_data, suffix, save_fpath=None):
    """Displays a histogram of average score per item.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    item_scores = {}
    for scores in score_data.values():
        for item_id, score in scores:
            item_scores.setdefault(item_id, []).append(score)
    avg_scores = {item_id: np.mean(scores) for item_id, scores in item_scores.items()}
    plt.hist(avg_scores.values(), bins="auto")
    plt.title("Histogram of Average Score per Item" + suffix)
    plt.xlabel("Average Score")
    plt.ylabel("Number of Items")
    display_fig(save_fpath)


def display_average_score_per_user_histogram(score_data, suffix, save_fpath=None):
    """Displays a histogram of average score per user.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
    """
    avg_scores = [
        np.mean([score for _, score in scores]) for scores in score_data.values()
    ]
    plt.hist(avg_scores, bins="auto")
    plt.title("Histogram of Average Score per User" + suffix)
    plt.xlabel("Average Score")
    plt.ylabel("Number of Users")
    display_fig(save_fpath)


def display_fig(save_fpath):
    if save_fpath:
        parent = os.path.dirname(save_fpath)
        os.makedirs(parent, exist_ok=True)
        plt.savefig(save_fpath, dpi=config.eda_figures_dpi)
    else:
        plt.show()


def display_all(score_data, user_data, suffix, fpath_base):
    get_score_range_and_counts(score_data)
    display_score_histogram(score_data, suffix, save_fpath=f"{fpath_base}/scores.png")
    display_score_count_per_item_histogram(
        score_data, suffix, save_fpath=f"{fpath_base}/score_counts_per_item.png"
    )
    display_average_score_per_item_histogram(
        score_data, suffix, save_fpath=f"{fpath_base}/average_score_per_item.png"
    )
    display_average_score_per_user_histogram(
        score_data, suffix, save_fpath=f"{fpath_base}/average_score_per_user.png"
    )


if __name__ == "__main__":
    # Load raw data
    score_data = data.get_score_data()
    user_data = data.get_user_data()

    print("Before data filtering:")
    suffix = " - before filtering"
    fpath_base = f"{config.eda_figures_dir_before_filtering}"
    display_all(score_data, user_data, suffix, fpath_base)

    # TODO: EDA for user data

    print("After data filtering:")
    suffix = " - after filtering"
    fpath_base = f"{config.eda_figures_dir_after_filering}"
    display_all(score_data, user_data, suffix, fpath_base)
