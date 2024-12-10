from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_basic_distributions(tweets_path: str, likes_path: str, output_path: str):
    """Plot simple distributions of the generated data"""
    # Load data
    likes_df = pd.read_csv(likes_path)
    tweets_df = pd.read_csv(tweets_path)

    # Set up the plots
    # plt.style.use("seaborn")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Distribution of likes per tweet (log scale)
    likes_per_tweet = likes_df["tweet_id"].value_counts()
    sns.histplot(likes_per_tweet, ax=axes[0, 0], log_scale=True)
    axes[0, 0].set_title("Distribution of Likes per Tweet")
    axes[0, 0].set_xlabel("Number of Likes (log scale)")
    axes[0, 0].set_ylabel("Count")

    # 2. Distribution of likes per user (log scale)
    likes_per_user = likes_df["user_id"].value_counts()
    sns.histplot(likes_per_user, ax=axes[0, 1], log_scale=True)
    axes[0, 1].set_title("Distribution of Activity per User")
    axes[0, 1].set_xlabel("Number of Likes Made (log scale)")
    axes[0, 1].set_ylabel("Count")

    # 3. Category distribution
    category_counts = pd.merge(
        likes_df, tweets_df[["id", "category"]], left_on="tweet_id", right_on="id"
    )["category"].value_counts()

    category_counts.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Likes by Category")
    axes[1, 0].set_xlabel("Category")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Basic stats
    stats_text = f"""
    Total Tweets: {len(tweets_df):,}
    Total Likes: {len(likes_df):,}
    Avg Likes per Tweet: {len(likes_df)/len(tweets_df):.1f}
    % Tweets with Likes: {100*len(likes_per_tweet)/len(tweets_df):.1f}%
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12)
    axes[1, 1].set_title("Basic Stats")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{output_path}/distributions.png")
    # plt.close()


if __name__ == "__main__":
    # Example usage
    dummy = str(Path(__file__).parent / "dummy_data")

    plot_basic_distributions(
        tweets_path=f"{dummy}/tweets.csv",
        likes_path=f"{dummy}/user_likes.csv",
        output_path=f"{dummy}",
    )
