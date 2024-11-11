import random
import keras

# TF is only used for tf.data - the code works with all backends
import tensorflow as tf

import numpy as np
import pandas as pd

from config import config


def get_raw_data():
    """Read raw interaction data, user data, and item data.

    THIS FUNCTION MUST BE CUSTOMIZED FOR THE TEMPLATE TO RUN
    ON YOUR OWN DATA.

    Returns:
        - interaction_data: List of dicts (on per interaction), such as:
            `[{"user": str, "item": str, "score": float}, ...]`
            `"score"` is optional.
        - user_data: Dict such as {"feature_name": value, ...}
        - item_data: Dict such as { "feature_name": value, ...}
    """
    df = pd.read_csv(config.csv_fpath)
    interaction_data = []
    user_data = {}
    item_data = {}
    for row in df.itertuples():
        items = row.no_seq_anonyme.split(" ")
        for item in items:
            interaction_data.append({"user": row.id_usager_anonyme, "item": item})
    return interaction_data, user_data, item_data


def filter_data(interaction_data, min_interactions_per_user, min_interactions_per_item):
    """Filter out items and users that have too few interactions.

    Args:
        interaction_data: List of dicts (on per interaction).
        min_interactions_per_user: Threshold below which to drop a user.
        min_interactions_per_item: Threshold below which to drop an item.

    Returns:
        New `interaction_data` list.
    """
    interaction_count_per_user = {}
    for event in interaction_data:
        user = event["user"]
        if user not in interaction_count_per_user:
            interaction_count_per_user[user] = 1
        else:
            interaction_count_per_user[user] += 1
    users_to_exclude = set(
        i
        for i, c in interaction_count_per_user.items()
        if c < min_interactions_per_user
    )
    interaction_data = [
        e for e in interaction_data if e["user"] not in users_to_exclude
    ]

    interaction_count_per_item = {}
    for event in interaction_data:
        item = event["item"]
        if item not in interaction_count_per_item:
            interaction_count_per_item[item] = 1
        else:
            interaction_count_per_item[item] += 1
    items_to_exclude = set(
        i
        for i, c in interaction_count_per_item.items()
        if c < min_interactions_per_item
    )
    return [e for e in interaction_data if e["item"] not in items_to_exclude]


def index_data(interaction_data):
    """Associates users and items with a unique, sequential integer ID.

    Returns:
        Tuple of two dicts, `user_index` and `item_index`.
        Each dict maps user/item names to a unique, sequential integer ID.
    """
    user_index = {}
    item_index = {}
    user_counter = 0
    item_counter = 0
    for event in interaction_data:
        user, item = event["user"], event["item"]
        if user not in user_index:
            user_index[user] = user_counter
            user_counter += 1
        if item not in item_index:
            item_index[item] = item_counter
            item_counter += 1
    return user_index, item_index


def filter_and_index_data(interaction_data, user_data=None, item_data=None):
    """Combines data filtering and indexing."""
    interaction_data = filter_data(
        interaction_data,
        min_interactions_per_user=config.min_interactions_per_user,
        min_interactions_per_item=config.min_interactions_per_item,
    )
    user_index, item_index = index_data(interaction_data)
    if user_data:
        user_data = {k: user_data[k] for k in user_index.keys()}
    if item_data:
        item_data = {k: item_data[k] for k in item_index.keys()}
    return interaction_data, user_index, item_index, user_data, item_data


def split_data(interaction_data):
    """Split interaction_data dict into training and validation data.

    Each user is associated with a number of interactions/items.
    We split those interactions into two subgroups: training interactions
    and validation interactions. Otherwise we would have
    users that are never seen at training time.

    Returns:
        Tuple of two lists, `train_data` and `val_data`.
        Each is a list of dicts structured like `interaction_data`.
    """
    train_data = []
    val_data = []
    total_events_per_user = {}
    total_val_event_per_user = {}
    val_event_per_user = {}
    for event in interaction_data:
        user = event["user"]
        if user not in total_events_per_user:
            total_events_per_user[user] = 1
        else:
            total_events_per_user[user] += 1
    for user, count in total_events_per_user.items():
        val_count = max(1, round(count * config.per_user_val_fraction))
        total_val_event_per_user[user] = val_count

    for event in interaction_data:
        user = event["user"]
        if user not in val_event_per_user:
            val_event_per_user[user] = 0
            val_events = 0
        else:
            val_events = val_event_per_user[user]
        if val_events < total_val_event_per_user[user]:
            val_data.append(event)
            val_event_per_user[user] += 1
        else:
            train_data.append(event)
    return train_data, val_data


def compute_baseline_scores(interaction_data):
    """Computes the prior probability that a user would interact
    with an item, if you know nothing about the user.

    Such probability scores tend to be very close to 0.

    Returns:
        A dict such as `{item: score}`.
    """
    score_sums = {}
    all_users = set()
    all_items = set()
    for e in interaction_data:
        item, user = e["item"], e["user"]
        all_users.add(user)
        all_items.add(item)
        if item not in score_sums:
            score_sums[item] = 0.0
        score_sums[item] += e.get("score", 1.0)
    return {item: score_sums[item] / len(all_users) for item in all_items}


def get_items_per_user(interaction_data):
    items_per_user = {}
    for e in interaction_data:
        user, item = e["user"], e["item"]
        if user not in items_per_user:
            items_per_user[user] = set()
        items_per_user[user].add(item)
    return items_per_user


class InteractionDataset(keras.utils.PyDataset):
    """Dataset that samples both positive interactions and negative ones.

    Yields:
        Tuple of batched inputs and targets `(x, y)`.
        - `x` is a dict with keys `"user_id"`, `"item_id"`,
        as well as optional keys `"user_features"`, `"item_features"`.
        The values they map to are batched arrays.
        - `y` is a batch of scalar score values, representing the interaction
        strength between a user and an item (between 0 and 1).

    If a user actually interacted with an item, the interaction score used
    is the known value of their rating of the item (if available) or 1 (default).
    If a user did not interact with an item, the interaction score used
    is the prior probability that the user would interact with the item if you
    know nothing about the user (which is close to 0).
    """

    def __init__(
        self,
        interaction_data,
        user_index,
        item_index,
        baseline_scores,
        user_features=None,
        item_features=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interaction_data = interaction_data
        self.user_index = user_index
        self.item_index = item_index
        self.all_users = list(user_index.keys())
        self.all_items = list(item_index.keys())
        self.baseline_scores = baseline_scores
        self.user_features = user_features
        self.item_features = item_features

        self.items_per_user = get_items_per_user(interaction_data)

        self.negative_sampling_fraction = config.negative_sampling_fraction
        self.batch_size = config.batch_size
        self.negative_samples_per_batch = round(
            self.batch_size * self.negative_sampling_fraction
        )

    @property
    def num_batches(self):
        # Pretty arbitrary since samples are drawn at random
        return round(len(self.interaction_data) * 2 / self.batch_size)

    def __getitem__(self, idx):
        # Return data for batch idx.
        batch = {
            "user_id": np.zeros(shape=(self.batch_size,), dtype="int32"),
            "item_id": np.zeros(shape=(self.batch_size,), dtype="int32"),
        }
        targets = np.zeros(shape=(self.batch_size, 1), dtype="int32")
        if self.user_features:
            user_features_shape = next(iter(self.user_features.values())).shape
            batch["user_features"] = np.zeros(
                shape=(self.batch_size,) + user_features_shape,
                dtype="float32",
            )
        if self.item_features:
            item_features_shape = next(iter(self.item_features.values())).shape
            batch["item_features"] = np.zeros(
                shape=(self.batch_size,) + item_features_shape,
                dtype="float32",
            )

        # Add negative samples.
        for i in range(self.negative_samples_per_batch):
            user = random.choice(self.all_users)
            item = random.choice(self.all_items)
            while item in self.items_per_user.get(user, ()):
                item = random.choice(self.all_items)
            batch["user_id"][i] = self.user_index[user]
            batch["item_id"][i] = self.item_index[item]
            if self.user_features:
                batch["user_features"][i] = self.user_features[user]
            if self.item_features:
                batch["item_features"][i] = self.item_features[item]
            targets[i] = self.baseline_scores[item]

        # Add position samples (actual interactions)
        for i in range(self.negative_samples_per_batch, self.batch_size):
            e = random.choice(self.interaction_data)
            user, item = e["user"], e["item"]
            batch["user_id"][i] = self.user_index[user]
            batch["item_id"][i] = self.item_index[item]
            if self.user_features:
                batch["user_features"][i] = self.user_features[user]
            if self.item_features:
                batch["item_features"][i] = self.item_features[item]
            targets[i] = e.get("score", 1.0)
        return batch, targets


def make_features_preprocessor(data, index, features_config):
    """Creates an adapt a Keras FeatureSpace to vectorize user features."""
    one_element = next(iter(data))
    reverse_index = {v: k for k, v in index.items()}
    features = {
        k: [data[reverse_index[i]][k] for i in range(len(index))]
        for k in data[one_element].keys()
    }
    preprocessor = keras.utils.FeatureSpace(
        features_config,
    )
    preprocessor.adapt(tf.data.Dataset.from_tensor_slices(features))
    return preprocessor


def apply_preprocessor(data, preprocessor):
    features = {}
    if data:
        for k, v in data.items():
            features[k] = preprocessor(v)
    return features


def get_preprocessed_features(user_data, item_data, user_index, item_index):
    """Turns user data and item data into preprocessed, vectorized dicts of features."""
    user_features = None
    if user_data:
        user_features_preprocessor = make_features_preprocessor(
            user_data, user_index, config.user_features_config
        )
        user_features = apply_preprocessor(user_data, user_features_preprocessor)
    item_features = None
    if item_data:
        item_features_preprocessor = make_features_preprocessor(
            item_data, item_index, config.item_features_config
        )
        item_features = apply_preprocessor(item_data, item_features_preprocessor)
    return user_features, item_features


def get_train_and_val_datasets(
    interaction_data, user_index, item_index, user_data, item_data
):
    """Creates tf.data.Dataset instances for the training data and the validation data."""
    train_data, val_data = split_data(interaction_data)
    baseline_scores = compute_baseline_scores(interaction_data)
    user_features, item_features = get_preprocessed_features(
        user_data, item_data, user_index, item_index
    )
    train_ds = InteractionDataset(
        train_data,
        user_index,
        item_index,
        baseline_scores=baseline_scores,
        user_features=user_features,
        item_features=item_features,
        use_multiprocessing=config.pydataset_use_multiprocessing,
        workers=config.pydataset_workers,
    )
    val_ds = InteractionDataset(
        val_data,
        user_index,
        item_index,
        baseline_scores=baseline_scores,
        user_features=user_features,
        item_features=item_features,
        use_multiprocessing=config.pydataset_use_multiprocessing,
        workers=config.pydataset_workers,
    )
    return train_ds, val_ds


def get_full_dataset(interaction_data, user_index, item_index, user_data, item_data):
    """Creates a tf.data.Dataset instance for the entire production data."""
    baseline_scores = compute_baseline_scores(interaction_data)
    user_features = None
    if user_data:
        user_features_preprocessor = make_features_preprocessor(
            user_data, user_index, config.user_features_config
        )
        user_features = apply_preprocessor(user_data, user_features_preprocessor)
    item_features = None
    if item_data:
        item_features_preprocessor = make_features_preprocessor(
            item_data, item_index, config.item_features_config
        )
        item_features = apply_preprocessor(item_data, item_features_preprocessor)
    return InteractionDataset(
        interaction_data,
        user_index,
        item_index,
        baseline_scores=baseline_scores,
        user_features=user_features,
        item_features=item_features,
        use_multiprocessing=config.pydataset_use_multiprocessing,
        workers=config.pydataset_workers,
    )
