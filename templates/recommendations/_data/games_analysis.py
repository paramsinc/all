from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_game_distributions(games_path: Path, ratings_path: Path, output_path: Path):
    """Plot distributions of the generated video game data"""
    # Load data
    ratings_df = pd.read_csv(ratings_path)
    games_df = pd.read_csv(games_path)

    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Distribution of ratings per game (log scale)
    ratings_per_game = ratings_df["game_id"].value_counts()
    sns.histplot(ratings_per_game, ax=axes[0, 0], log_scale=True)
    axes[0, 0].set_title("Distribution of Ratings per Game")
    axes[0, 0].set_xlabel("Number of Ratings (log scale)")
    axes[0, 0].set_ylabel("Count")

    # 2. Distribution of ratings per user (log scale)
    ratings_per_user = ratings_df["user_id"].value_counts()
    sns.histplot(ratings_per_user, ax=axes[0, 1], log_scale=True)
    axes[0, 1].set_title("Distribution of Activity per User")
    axes[0, 1].set_xlabel("Number of Ratings Made (log scale)")
    axes[0, 1].set_ylabel("Count")

    # 3. Category and rating distribution
    merged_df = pd.merge(
        ratings_df,
        games_df[["id", "category", "difficulty"]],
        left_on="game_id",
        right_on="id",
    )

    # Calculate average rating by category
    avg_ratings = (
        merged_df.groupby("category")["rating"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
    )

    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    # Plot bars for count
    avg_ratings["count"].plot(kind="bar", ax=ax3, color="skyblue", alpha=0.5)
    ax3.set_ylabel("Number of Ratings", color="skyblue")

    # Plot line for average rating
    avg_ratings["mean"].plot(kind="line", marker="o", ax=ax3_twin, color="red")
    ax3_twin.set_ylabel("Average Rating", color="red")

    ax3.set_title("Ratings by Category")
    ax3.set_xlabel("Category")
    ax3.tick_params(axis="x", rotation=45)

    # 4. Basic stats and difficulty distribution
    # Calculate difficulty distribution
    difficulty_dist = merged_df["difficulty"].value_counts()

    stats_text = f"""
    Total Games: {len(games_df):,}
    Total Ratings: {len(ratings_df):,}
    Avg Ratings per Game: {len(ratings_df)/len(games_df):.1f}
    Overall Avg Rating: {ratings_df['rating'].mean():.2f}
    
    Difficulty Distribution:
    {difficulty_dist.to_string()}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12)
    axes[1, 1].set_title("Basic Stats")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
    # plt.savefig(Path(output_path) / "game_distributions.png")
    # plt.close()


def plot_detailed_analysis(
    games_path: Path, ratings_path: Path, users_path: Path, output_path: Path
):
    """Create additional plots showing demographic analysis"""
    # Load all data
    ratings_df = pd.read_csv(ratings_path)
    games_df = pd.read_csv(games_path)
    users_df = pd.read_csv(users_path)

    # Merge data
    merged_df = pd.merge(
        ratings_df,
        games_df[["id", "category", "difficulty"]],
        left_on="game_id",
        right_on="id",
    )

    merged_df = pd.merge(
        merged_df, users_df[["id", "age", "gender"]], left_on="user_id", right_on="id"
    )

    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Average rating by age group and difficulty
    merged_df["age_group"] = pd.cut(
        merged_df["age"],
        bins=[0, 20, 30, 40, 50, 100],
        labels=["<20", "20-30", "30-40", "40-50", "50+"],
    )

    pivot_diff = merged_df.pivot_table(
        values="rating", index="age_group", columns="difficulty", aggfunc="mean"
    )

    pivot_diff.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Average Rating by Age Group and Difficulty")
    axes[0, 0].set_xlabel("Age Group")
    axes[0, 0].set_ylabel("Average Rating")
    axes[0, 0].legend(title="Difficulty")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Category preference by gender
    pivot_gender = merged_df.pivot_table(
        values="rating", index="category", columns="gender", aggfunc="mean"
    )

    pivot_gender.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Average Rating by Category and Gender")
    axes[0, 1].set_xlabel("Category")
    axes[0, 1].set_ylabel("Average Rating")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Rating distribution by difficulty
    sns.boxplot(data=merged_df, x="difficulty", y="rating", ax=axes[1, 0])
    axes[1, 0].set_title("Rating Distribution by Difficulty")
    axes[1, 0].set_xlabel("Difficulty")
    axes[1, 0].set_ylabel("Rating")

    # 4. Average rating over time
    merged_df["date"] = pd.to_datetime(merged_df["timestamp_ms"], unit="ms")
    time_ratings = merged_df.set_index("date")["rating"].resample("W").mean()

    time_ratings.plot(ax=axes[1, 1])
    axes[1, 1].set_title("Average Rating Over Time")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Average Rating")

    plt.tight_layout()
    plt.show()
    # plt.savefig(Path(output_path) / "detailed_analysis.png")
    # plt.close()


if __name__ == "__main__":
    # Example usage
    dummy = Path(__file__).parent / "dummy_data"

    plot_game_distributions(
        games_path=dummy / "games.csv",
        ratings_path=dummy / "user_ratings.csv",
        output_path=dummy,
    )

    plot_detailed_analysis(
        games_path=dummy / "games.csv",
        ratings_path=dummy / "user_ratings.csv",
        users_path=dummy / "users.csv",
        output_path=dummy,
    )
