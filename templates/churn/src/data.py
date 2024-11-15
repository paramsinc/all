import os
import typing as T
from datetime import datetime, timedelta

import joblib
import numpy as np
import polars as pl
from keras import utils
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.config import config

event_dtype = pl.Enum(
    [
        "new membership",
        "rejoin membership",
        "renewed membership",
        "wash",
        "expired",
        "terminated",
    ]
)

# leave 0 for padding
event_type_to_int = {
    "new membership": 1,
    "rejoin membership": 2,
    "renewed membership": 3,
    "wash": 4,
    "expired": 5,
    "terminated": 6,
}


def read_in_data(limit: int = None) -> pl.DataFrame:
    """Read in the data from the csv files and return a DataFrame like this:
    ┌────────────────┬─────────────────────────────────┬─────────────────────────────────┐
    │ membership_id  ┆ timestamp                       ┆ event_type                      │
    │ ---            ┆ ---                             ┆ ---                             │
    │ str            ┆ list[datetime[μs]]              ┆ list[enum]                      │
    ╞════════════════╪═════════════════════════════════╪═════════════════════════════════╡
    │ 001            ┆ [2021-09-17 16:58:00, 2021-08-… ┆ ["wash", "wash", … "renewed me… │
    │ 002            ┆ [2022-02-06 11:36:00, 2022-03-… ┆ ["wash", "wash", … "renewed me… │
    │ 003            ┆ [2023-05-02 14:49:00, 2023-02-… ┆ ["wash", "wash", … "renewed me… │
    │ 004            ┆ [2021-12-20 00:00:00]           ┆ ["expired"]                     │
    │ 005            ┆ [2024-09-07 11:49:00, 2024-09-… ┆ ["wash", "wash", … "new member… │
    └────────────────┴─────────────────────────────────┴─────────────────────────────────┘
    """
    washes_df = (
        pl.read_csv(
            config.data_path + "raw_transactions.csv",
            columns=["MEMBERSHIP_ID", "CREATED_AT"],
            schema_overrides={
                "MEMBERSHIP_ID": pl.Utf8,
                "CREATED_AT": pl.Utf8,
            },
        )
        .with_columns(pl.col("CREATED_AT").str.to_datetime(format="%m/%d/%Y %I:%M %p"))
        .with_columns(event_type=pl.lit("wash", dtype=event_dtype))
        .rename({"MEMBERSHIP_ID": "membership_id", "CREATED_AT": "timestamp"})
    )

    churn_df = (
        pl.read_csv(
            config.data_path + "raw_churn_events.csv",
            columns=["MEMBERSHIP_ID", "CHURN_DATE", "CHURN_TYPE"],
            schema_overrides={
                "MEMBERSHIP_ID": pl.Utf8,
                "CHURN_DATE": pl.Utf8,
                "CHURN_TYPE": pl.Utf8,
            },
        )
        .with_columns(pl.col("CHURN_DATE").str.to_datetime(format="%m/%d/%Y"))
        .with_columns(pl.col("CHURN_TYPE").cast(event_dtype))
        .rename(
            {
                "MEMBERSHIP_ID": "membership_id",
                "CHURN_DATE": "timestamp",
                "CHURN_TYPE": "event_type",
            }
        )
    )

    membership_df = (
        pl.read_csv(
            config.data_path + "raw_memberships.csv",
            columns=["MEMBERSHIP_ID", "CREATED_AT", "TRANSACTION_CATEGORY"],
            schema_overrides={
                "MEMBERSHIP_ID": pl.Utf8,
                "CREATED_AT": pl.Utf8,
                "TRANSACTION_CATEGORY": pl.Utf8,
            },
        )
        .with_columns(pl.col("CREATED_AT").str.to_datetime(format="%m/%d/%Y %I:%M %p"))
        .with_columns(pl.col("TRANSACTION_CATEGORY").cast(event_dtype))
        .rename(
            {
                "MEMBERSHIP_ID": "membership_id",
                "CREATED_AT": "timestamp",
                "TRANSACTION_CATEGORY": "event_type",
            }
        )
    )

    df = pl.concat([washes_df, churn_df, membership_df])
    grouped_df = df.group_by("membership_id").agg(
        pl.col("timestamp"), pl.col("event_type")
    )
    if limit:
        grouped_df = grouped_df.head(limit)
    return grouped_df


def transform_to_sorted_dict(
    grouped_df: pl.DataFrame,
) -> dict[str, list[dict[str, T.Any]]]:
    """_summary_
    Transform the grouped DataFrame into the desired dictionary with a progress bar and sorted by timestamp

    Args:
        grouped_df (pl.DataFrame): _description_

    Returns:
        dict[str, list[dict[str, T.Any]]]: _description_

    Example:
    {
        "oo1": [
            {
                'timestamp': datetime.datetime(2022, 3, 25, 10, 24),
                'event_type': 'wash',
            },
            {
                'timestamp': datetime.datetime(2022, 5, 3, 0, 0),
                'event_type': 'expired',
            },
        ]
    }
    """
    membership_dict = {
        row["membership_id"]: sorted(
            [
                {"timestamp": ts, "event_type": et}
                for ts, et in zip(row["timestamp"], row["event_type"], strict=False)
            ],
            key=lambda event: event["timestamp"],
        )
        for row in tqdm(grouped_df.to_dicts(), desc="Processing Memberships")
    }
    return membership_dict


def get_last_timestamp(membership_dict: dict[str, list[dict[str, T.Any]]]) -> datetime:
    """Returns the last timestamp in the data."""
    last_timestamp: datetime = datetime(2000, 1, 1)

    for events in membership_dict.values():
        last_event = events[-1]
        last_timestamp = max(last_event["timestamp"], last_timestamp)

    return last_timestamp


def tag_membership_ids(
    membership_dict: dict[str, list[dict[str, T.Any]]],
) -> dict[str, set[str]]:
    # tag membership_id as churned or not... exclude the ones that have been terminated
    tagged: dict[str, set[str]] = {
        "active": set(),
        "expired": set(),
        "terminated": set(),
    }

    for membership_id, events in membership_dict.items():
        last_event = events[-1]
        if last_event["event_type"] == "expired":
            tagged["expired"].add(membership_id)
        elif last_event["event_type"] == "terminated":
            tagged["terminated"].add(membership_id)
        else:
            tagged["active"].add(membership_id)

    print(
        f"len active memberships: {len(tagged['active'])}, len expired memberships: {len(tagged['expired'])}, len terminated memberships: {len(tagged['terminated'])}"
    )
    return tagged


def build_event_sequences_time_differences_and_labels(
    membership_dict: dict[str, list[dict[str, T.Any]]],
    tagged_membership_ids: dict[str, set[str]],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    sequences: list[list[int]] = []
    time_differences: list[list[int]] = []
    labels: list[int] = []

    last_timestamp = get_last_timestamp(membership_dict=membership_dict)

    filter_non_churners_ts = last_timestamp - timedelta(days=config.churn_days_out)
    print(filter_non_churners_ts)

    for membership_id, events in tqdm(
        membership_dict.items(), desc="Processing Features"
    ):
        # Skip expired memberships
        if membership_id in tagged_membership_ids["expired"]:
            continue

        if membership_id in tagged_membership_ids["terminated"]:
            # remove the last item in the sequence which would be "terminated"
            events = events[:-1]
            label = 1
        else:
            label = 0
            # filter out for non churners
            events = [e for e in events if e["timestamp"] < filter_non_churners_ts]

        if not events:
            continue

        # if there are not 20 events, just filter it out... too much noise
        if len(events) < config.min_events:
            continue

        # Extract the sequence of event types
        event_sequence = [event_type_to_int[event["event_type"]] for event in events]

        # Calculate time differences between events
        timestamps = [event["timestamp"] for event in events]
        time_diff_sequence: list[int] = []
        for i in range(1, len(timestamps)):
            diff = int((timestamps[i] - timestamps[i - 1]).total_seconds())
            time_diff_sequence.append(diff)
        # For the first event, insert a default time difference (e.g., 0)
        time_diff_sequence.insert(0, 0)

        sequences.append(event_sequence)
        time_differences.append(time_diff_sequence)
        labels.append(label)
    return sequences, time_differences, labels


def fit_and_save_scaler(time_differences: list[list[int]]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on time differences and save it for later use.

    Args:
        time_differences: List of lists containing time difference sequences

    Returns:
        Fitted MinMaxScaler object
    """
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Flatten the list of time differences to fit the scaler
    flattened_time_diffs = np.concatenate(time_differences)

    # Fit the scaler on the flattened array
    scaler.fit(flattened_time_diffs.reshape(-1, 1))

    # Save the fitted scaler
    os.makedirs(os.path.dirname(config.time_diff_scaler_path), exist_ok=True)
    print(f"Saving scaler to {config.time_diff_scaler_path}")
    joblib.dump(scaler, config.time_diff_scaler_path)

    return scaler


def get_scaler() -> MinMaxScaler:
    return joblib.load(config.time_diff_scaler_path)


def normalize_time_differences(
    time_differences: list[list[int]], scaler: MinMaxScaler
) -> list[list[float]]:
    """
    Normalize time differences using a MinMaxScaler.

    Args:
        time_differences: List of lists containing time difference sequences
        scaler: Optional pre-fitted MinMaxScaler

    Returns:
        List of normalized time difference sequences
    """

    # Normalize each sequence
    normalized_time_differences = [
        scaler.transform(np.array(seq).reshape(-1, 1)).flatten()
        for seq in tqdm(time_differences, desc="Normalizing Time Diffs")
    ]

    return normalized_time_differences


def get_padded_sequences_and_time_diffs_and_labels(
    sequences: list[list[int]],
    normalized_time_differences: list[list[float]],
    labels: list[int],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    # since transformers must take in embeddings of the same dimension, we need to pad the inputs to the same dimension

    # Pad event sequences
    padded_sequences = utils.pad_sequences(
        sequences,
        maxlen=config.max_embedding_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="int16",
    )

    # Pad time difference sequences
    padded_time_diffs = utils.pad_sequences(
        normalized_time_differences,
        maxlen=config.max_embedding_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="float16",
    )

    # Convert labels to a NumPy array
    labels = np.array(labels)
    return padded_sequences, padded_time_diffs, labels


def get_padded_sequences_and_time_diffs_and_labels_from_config(
    limit: int = None,
) -> tuple[list[list[int]], list[list[float]], list[int]]:
    grouped_df = read_in_data(limit=limit)
    membership_dict = transform_to_sorted_dict(grouped_df=grouped_df)
    tagged_membership_ids = tag_membership_ids(membership_dict=membership_dict)
    sequences, time_differences, labels = (
        build_event_sequences_time_differences_and_labels(
            membership_dict=membership_dict, tagged_membership_ids=tagged_membership_ids
        )
    )
    scaler = fit_and_save_scaler(time_differences=time_differences)
    normalized_time_differences = normalize_time_differences(
        time_differences=time_differences, scaler=scaler
    )

    return get_padded_sequences_and_time_diffs_and_labels(
        sequences=sequences,
        normalized_time_differences=normalized_time_differences,
        labels=labels,
    )
