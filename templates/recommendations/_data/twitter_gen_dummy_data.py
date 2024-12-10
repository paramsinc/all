import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    # Data size configuration
    N_USERS: int = 10_000
    N_TWEETS: int = 100_000
    N_LIKES: int = 1_000_000
    TIME_PERIOD_DAYS: int = 90

    # User generation parameters
    MIN_AGE: int = 13
    MAX_AGE: int = 80
    MEAN_AGE: int = 30
    AGE_STD: int = 10
    GENDERS: list[str] = field(default_factory=lambda: ["M", "F"])

    # Tweet categories
    CATEGORIES: list[str] = field(
        default_factory=lambda: [
            "news",
            "sports",
            "gaming",
            "music",
            "movies",
            "technology",
            "politics",
            "food",
            "travel",
        ]
    )

    # Tweet length distributions (mean, std)
    LENGTH_DISTRIBUTIONS: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "news": (200, 50),
            "sports": (100, 30),
            "gaming": (80, 20),
            "music": (120, 40),
            "movies": (150, 40),
            "technology": (180, 50),
            "politics": (220, 60),
            "food": (130, 30),
            "travel": (200, 50),
        }
    )

    # Tweet length constraints
    MIN_TWEET_LENGTH: int = 30
    MAX_TWEET_LENGTH: int = 280

    # Engagement parameters
    TIME_DECAY_HOURS: float = 72.0
    VIRAL_COEFFICIENT: float = 0.15

    # Age-based thresholds
    YOUNG_AGE_THRESHOLD: int = 25
    ADULT_AGE_THRESHOLD: int = 40

    # Category preference multipliers
    AGE_CATEGORY_PREFS: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "young": {
                "gaming": 1.5,
                "news": 0.7,
                "politics": 0.6,
                "music": 1.3,
                "movies": 1.2,
                "technology": 1.1,
                "sports": 1.0,
                "food": 0.8,
                "travel": 0.9,
            },
            "adult": {
                "gaming": 0.5,
                "news": 1.3,
                "politics": 1.4,
                "music": 0.8,
                "movies": 0.9,
                "technology": 0.9,
                "sports": 1.0,
                "food": 1.2,
                "travel": 1.1,
            },
        }
    )

    # Gender-based category preferences
    GENDER_CATEGORY_PREFS: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "M": {"sports": 1.2, "gaming": 1.1, "technology": 1.1},
            "F": {"food": 1.2, "travel": 1.1, "music": 1.1},
        }
    )

    # Length preference multipliers
    LENGTH_PREF_MULTIPLIERS: dict[str, float] = field(
        default_factory=lambda: {
            "short_young": 1.3,  # Young users with short tweets
            "long_young": 0.7,  # Young users with long tweets
            "short_adult": 0.7,  # Adult users with short tweets
            "long_adult": 1.3,  # Adult users with long tweets
        }
    )

    # Sampling parameters
    MAX_TWEETS_TO_SAMPLE: int = 100


@dataclass
class User:
    id: UUID
    age: int
    gender: str


@dataclass
class Tweet:
    id: UUID
    created_at_ms: int
    category: str
    length: int


@dataclass
class Like:
    user_id: UUID
    tweet_id: UUID
    timestamp_ms: int


class TweetDataGenerator:
    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        self.config = config
        self.end_date: datetime = datetime.now()
        self.start_date: datetime = self.end_date - timedelta(
            days=config.TIME_PERIOD_DAYS
        )

    def generate_users(self) -> pd.DataFrame:
        users: list[User] = []
        for _ in range(self.config.N_USERS):
            age = int(np.random.normal(self.config.MEAN_AGE, self.config.AGE_STD))
            age = max(self.config.MIN_AGE, min(age, self.config.MAX_AGE))
            users.append(
                User(id=uuid4(), age=age, gender=random.choice(self.config.GENDERS))
            )
        return pd.DataFrame([vars(u) for u in users])

    def generate_tweets(self) -> pd.DataFrame:
        tweets: list[Tweet] = []
        for _ in range(self.config.N_TWEETS):
            category = random.choice(self.config.CATEGORIES)
            mean_length, std_length = self.config.LENGTH_DISTRIBUTIONS[category]
            length = int(np.random.normal(mean_length, std_length))
            length = max(
                self.config.MIN_TWEET_LENGTH, min(length, self.config.MAX_TWEET_LENGTH)
            )

            created_at = self.start_date + timedelta(
                seconds=random.randint(
                    0, int((self.end_date - self.start_date).total_seconds())
                )
            )

            tweets.append(
                Tweet(
                    id=uuid4(),
                    created_at_ms=int(created_at.timestamp() * 1000),
                    category=category,
                    length=length,
                )
            )
        return pd.DataFrame([vars(t) for t in tweets])

    def calculate_tweet_score(
        self,
        tweet: dict,
        user: dict,
        current_time: float,
        viral_scores: dict[UUID, int],
    ) -> float:
        # Time decay factor
        time_diff_hours = (current_time - tweet["created_at_ms"] / 1000) / 3600
        time_factor = np.exp(-time_diff_hours / self.config.TIME_DECAY_HOURS)

        # Age-based category preferences
        age_group = (
            "young" if user["age"] < self.config.YOUNG_AGE_THRESHOLD else "adult"
        )
        category_factor = self.config.AGE_CATEGORY_PREFS[age_group][tweet["category"]]

        # Gender-based preferences
        gender_factor = self.config.GENDER_CATEGORY_PREFS.get(user["gender"], {}).get(
            tweet["category"], 1.0
        )

        # Length preference based on age
        length_key = f"{'short' if tweet['length'] < 100 else 'long'}_{age_group}"
        length_factor = self.config.LENGTH_PREF_MULTIPLIERS[length_key]

        # Viral factor
        viral_factor = 1.0 + (
            viral_scores.get(tweet["id"], 0) * self.config.VIRAL_COEFFICIENT
        )

        return (
            time_factor * category_factor * gender_factor * length_factor * viral_factor
        )

    def generate_user_likes(
        self, users_df: pd.DataFrame, tweets_df: pd.DataFrame
    ) -> pd.DataFrame:
        start_time = min(tweets_df["created_at_ms"]) // 1000
        end_time = max(tweets_df["created_at_ms"]) // 1000

        likes: list[Like] = []
        viral_scores: dict[UUID, int] = {}

        # Modified dictionary creation to include id in the values
        users_dict = users_df.reset_index().to_dict("records")
        tweets_dict = tweets_df.reset_index().to_dict("records")

        # Create lookup dictionaries by ID
        users_by_id = {str(u["id"]): u for u in users_dict}
        tweets_by_id = {str(t["id"]): t for t in tweets_dict}

        current_users = list(users_by_id.keys())
        current_tweets = list(tweets_by_id.keys())

        n_likes = self.config.N_LIKES

        for i, current_time in enumerate(np.linspace(start_time, end_time, n_likes)):
            if i % 1000 == 0:
                print(f"Processing {i + 1}/{n_likes}")
            user_id = random.choice(current_users)
            user = users_by_id[user_id]

            available_tweets = [
                tweet_id
                for tweet_id in current_tweets
                if tweets_by_id[tweet_id]["created_at_ms"] / 1000 <= current_time
            ]

            tweet_scores: list[tuple[str, float]] = []
            sample_size = min(self.config.MAX_TWEETS_TO_SAMPLE, len(available_tweets))

            for tweet_id in random.sample(available_tweets, sample_size):
                score = self.calculate_tweet_score(
                    tweets_by_id[tweet_id], user, current_time, viral_scores
                )
                tweet_scores.append((tweet_id, score))

            if tweet_scores:
                total_score = sum(score for _, score in tweet_scores)
                if total_score > 0:
                    selected_tweet_id = random.choices(
                        [t[0] for t in tweet_scores],
                        weights=[t[1] / total_score for t in tweet_scores],
                        k=1,
                    )[0]

                    viral_scores[selected_tweet_id] = (
                        viral_scores.get(selected_tweet_id, 0) + 1
                    )

                    likes.append(
                        Like(
                            user_id=UUID(user_id),
                            tweet_id=UUID(selected_tweet_id),
                            timestamp_ms=int(current_time * 1000),
                        )
                    )

        return pd.DataFrame([vars(l) for l in likes])

    def generate_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Generating users...")
        users_df = self.generate_users()

        print("Generating tweets...")
        tweets_df = self.generate_tweets()

        print("Generating likes...")
        likes_df = self.generate_user_likes(users_df, tweets_df)

        # get current file path using Path
        dummy = Path(__file__).parent / "dummy_data"

        # Save to CSV
        users_df.to_csv(dummy / "users.csv", index=False)
        tweets_df.to_csv(dummy / "tweets.csv", index=False)
        likes_df.to_csv(dummy / "user_likes.csv", index=False)

        return users_df, tweets_df, likes_df


def main():
    # Example of using custom config
    # custom_config = GeneratorConfig(
    #     N_USERS=5000,  # Smaller dataset
    #     N_TWEETS=50000,
    #     N_LIKES=500000,
    #     TIME_PERIOD_DAYS=30,  # One month of data
    #     VIRAL_COEFFICIENT=0.2,  # Stronger viral effect
    # )

    generator = TweetDataGenerator(
        GeneratorConfig(
            N_LIKES=100_000,
            N_TWEETS=10_000,
            N_USERS=1_000,
            TIME_PERIOD_DAYS=30,
            VIRAL_COEFFICIENT=0.2,
        )
    )
    generator.generate_all_data()


if __name__ == "__main__":
    main()
