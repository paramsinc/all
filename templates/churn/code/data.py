# %%
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from keras import utils
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

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

# TODO wget
path = ""

washes_df = (
    pl.read_csv(
        path + "raw_transactions.csv",
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
        path + "raw_churn_events.csv",
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
        path + "raw_memberships.csv",
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

# %%

grouped_df = df.group_by("membership_id").agg(pl.col("timestamp"), pl.col("event_type"))

# %%

# Transform the grouped DataFrame into the desired dictionary with a progress bar and sorted by timestamp
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

# %%

# tag membership_id as churned or not... exclude the ones that have been terminated
active_membership_ids: set[str] = set()
expired_membership_ids: set[str] = set()
terminated_membership_ids: set[str] = set()

last_timestamp: datetime | None = datetime(2000, 1, 1)

for membership_id, events in membership_dict.items():
    last_event = events[-1]
    last_timestamp = max(last_event["timestamp"], last_timestamp)
    if last_event["event_type"] == "expired":
        expired_membership_ids.add(membership_id)
    elif last_event["event_type"] == "terminated":
        terminated_membership_ids.add(membership_id)
    else:
        active_membership_ids.add(membership_id)

print(
    f"len active memberships: {len(active_membership_ids)}, len expired memberships: {len(expired_membership_ids)}, len terminated memberships: {len(terminated_membership_ids)}"
)

# %%
# leave 0 for padding
event_type_to_int = {
    "new membership": 1,
    "rejoin membership": 2,
    "renewed membership": 3,
    "wash": 4,
    "expired": 5,
    "terminated": 6,
}

sequences: list[list[int]] = []
time_differences: list[list[int]] = []
labels: list[int] = []

# CHURN_DAYS_OUT = 90
CHURN_DAYS_OUT = 30
MIN_EVENTS = 20

filter_non_churners_ts = last_timestamp - timedelta(days=CHURN_DAYS_OUT)
print(filter_non_churners_ts)

for membership_id, events in tqdm(membership_dict.items(), desc="Processing Features"):
    # Skip expired memberships
    if membership_id in expired_membership_ids:
        continue

    if membership_id in terminated_membership_ids:
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
    if len(events) < MIN_EVENTS:
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

# %%
# normalize time differences
# Initialize the scaler
scaler = MinMaxScaler()

# Flatten the list of time differences to fit the scaler
flattened_time_diffs = np.concatenate(time_differences)

# Fit the scaler on the flattened array
scaler.fit(flattened_time_diffs.reshape(-1, 1))

# Normalize each sequence
normalized_time_differences = [
    scaler.transform(np.array(seq).reshape(-1, 1)).flatten()
    for seq in tqdm(time_differences, desc="Normalizing Time Diffs")
]

# %%

# since transformers must take in embeddings of the same dimension, we need to pad the inputs to the same dimension
max_sequence_length = max(len(seq) for seq in sequences)
# max time diff will be the same as max seq but just for clarity:
max_time_diff_length = max(len(i) for i in normalized_time_differences)

MAX_EMBEDDING_LEN = 150
# MAX_EMBEDDING_LEN = 50

# choose a reasonable max sequence length and time diff length
max_sequence_length = min(MAX_EMBEDDING_LEN, max_sequence_length)
max_time_diff_length = min(MAX_EMBEDDING_LEN, max_time_diff_length)

# Pad event sequences
padded_sequences = utils.pad_sequences(
    sequences,
    maxlen=max_sequence_length,
    padding="pre",
    truncating="pre",
    value=0,
    dtype="int16",
)

# Pad time difference sequences
padded_time_diffs = utils.pad_sequences(
    normalized_time_differences,
    maxlen=max_time_diff_length,
    padding="pre",
    truncating="pre",
    value=0,
    dtype="float16",
)

# Convert labels to a NumPy array
labels = np.array(labels)
