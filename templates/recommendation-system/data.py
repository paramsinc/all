import random
import keras

# TF is only used for tf.data - the code works with all backends
import tensorflow as tf

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from config import config


def get_score_data():
    """Return dict structured as {user_id: [(item_id, score), ...], ...}"""
    df = pd.read_parquet(config.score_data_parquet_fpath)
    score_data = {}
    for row in df.itertuples():
        if row.ref_player not in score_data:
            score_data[row.ref_player] = []
        score_data[row.ref_player].append((row.ref_game, row.rating))
    return score_data


def get_user_data():
    """Return dict structured as {user_id: {feature_name: value, ...}, ...]}"""
    score_data = get_score_data()
    user_data = {}
    age_choices = list(range(21, 90))
    gender_choices = ["male", "female", "unknown"]
    for key in score_data.keys():
        user_data[key] = {
            "age": random.choice(age_choices),
            "gender": random.choice(gender_choices),
        }
    return user_data


def filter_score_data(score_data, min_scores_per_user, min_scores_per_item):
    """Filter out items that have too few scores.

    Also proceededs to filter out users that subsequently have too few
    items scored.

    Args:
        score_data: Dict `{user_id: [(item_id, score), ...], ...}`
        min_scores_per_user: Threshold below which to drop a user.
        min_scores_per_item: Threshold below which to drop an item.

    Returns:
        New `score_data` dict.
    """
    score_data = {u: s for u, s in score_data.items() if len(s) >= min_scores_per_user}

    score_count_per_item = {}
    for user, scores in score_data.items():
        for item, _ in scores:
            if item not in score_count_per_item:
                score_count_per_item[item] = 1
            else:
                score_count_per_item[item] += 1
    items_to_exclude = set(
        i for i, c in score_count_per_item.items() if c < min_scores_per_item
    )

    new_score_data = {}
    for user, scores in score_data.items():
        new_scores = []
        for item, score in scores:
            if item not in items_to_exclude:
                new_scores.append((item, score))
        if len(new_scores) >= min_scores_per_user:
            new_score_data[user] = new_scores
    return new_score_data


def split_score_data_into_inputs_and_targets(score_data):
    """Split the score_data dict into input scores and target scores.

    Each user is associated with a number of scores.
    We split those scores into two subgroups: input scores
    and target scores. The idea is to show the model
    the input scores, predict scores for all items,
    and only train/eval based on the target scores.
    """
    input_score_data = {}
    target_score_data = {}
    for user, scores in score_data.items():
        num_to_drop = max(1, round(len(scores) * config.target_score_fraction))
        random.shuffle(scores)
        inputs_scores = scores[:-num_to_drop]
        targets_scores = scores[-num_to_drop:]
        input_score_data[user] = inputs_scores
        target_score_data[user] = targets_scores
    return input_score_data, target_score_data


def index_users_and_items(score_data):
    """Associates users and items with a unique integer ID."""
    user_to_id = {}
    item_to_id = {}
    user_index = 0
    item_index = 0
    for user, scores in score_data.items():
        if user not in user_to_id:
            user_to_id[user] = user_index
            user_index += 1
        for item, _ in scores:
            if item not in item_to_id:
                item_to_id[item] = item_index
                item_index += 1
    return user_to_id, item_to_id


def vectorize_score_data(
    score_data, user_to_id, item_to_id, sparse=True, dtype="float32"
):
    """Split score data into inputs and targets and turn them into sparse (or dense) matrices."""
    input_score_data, target_score_data = split_score_data_into_inputs_and_targets(
        score_data
    )
    input_matrix = make_score_matrix(
        input_score_data, user_to_id, item_to_id, sparse=sparse, dtype=dtype
    )
    target_matrix = make_score_matrix(
        target_score_data, user_to_id, item_to_id, sparse=sparse, dtype=dtype
    )
    return input_matrix, target_matrix


def make_score_matrix(score_data, user_to_id, item_to_id, sparse=True, dtype="float32"):
    """Turns score data into a sparse (or dense) matrix."""
    shape = (len(score_data), len(item_to_id))

    if sparse:
        matrix = lil_matrix(shape, dtype=dtype)
    else:
        matrix = np.zeros(shape, dtype=dtype)
    for user, scores in score_data.items():
        user_id = user_to_id[user]
        for item, score in scores:
            item_id = item_to_id.get(item, None)
            if item_id is not None:
                matrix[user_id, item_id] = score
    return matrix


def sparse_matrix_to_dataset(sparse_matrix):
    """Turn a sparse matrix into a tf.data.Dataset."""
    coo_matrix = sparse_matrix.tocoo()
    indices = np.vstack((coo_matrix.row, coo_matrix.col)).transpose()
    sparse_tensor = tf.SparseTensor(
        indices=indices, values=coo_matrix.data, dense_shape=sparse_matrix.shape
    )
    ds = tf.data.Dataset.from_tensor_slices((sparse_tensor,))
    return ds.map(lambda x: tf.sparse.to_dense(x))


def scale_score_matrix(score_matrix):
    if config.score_scaling_factor is not None:
        return score_matrix / config.score_scaling_factor
    return score_matrix


def make_dataset(
    input_scores, target_scores, user_features, user_features_preprocessor, batch_size
):
    """Turn score and user data a into tf.data.Dataset."""
    if isinstance(input_scores, lil_matrix):
        input_scores_ds = sparse_matrix_to_dataset(input_scores)
    else:
        input_scores_ds = tf.data.Dataset.from_tensor_slices((input_scores,))
    if isinstance(target_scores, lil_matrix):
        target_scores_ds = sparse_matrix_to_dataset(target_scores)
    else:
        target_scores_ds = tf.data.Dataset.from_tensor_slices((target_scores,))

    features_ds = tf.data.Dataset.from_tensor_slices((user_features,))
    features_ds = features_ds.map(user_features_preprocessor, num_parallel_calls=8)
    # dataset = tf.data.Dataset.zip(input_scores_ds, target_scores_ds, features_ds)
    dataset = tf.data.Dataset.zip(input_scores_ds, target_scores_ds)
    # dataset = dataset.map(
    #     lambda x, y, z: (tf.concat((x, z), axis=-1), y), num_parallel_calls=8
    # )
    return dataset.batch(batch_size).prefetch(8)


def prepare_user_features(user_data, user_to_id):
    """Turns user data into the format
    {feature_name: [value_for_user_0, value_for_user_1, ...], ...}"""
    one_user = next(iter(user_data))
    ids = range(len(user_to_id))
    id_to_user = {v: k for k, v in user_to_id.items()}
    user_features = {
        k: [user_data[id_to_user[i]][k] for i in ids]
        for k in user_data[one_user].keys()
    }
    return user_features


def make_user_features_preprocessor(user_features, feature_config):
    """Creates an adapt a Keras FeatureSpace to vectorize user features."""
    preprocessor = keras.utils.FeatureSpace(
        feature_config,
    )
    preprocessor.adapt(tf.data.Dataset.from_tensor_slices(user_features))
    return preprocessor


def filter_and_index_data(score_data, user_data):
    """Filters out users and items with insufficient score data,
    and computes integer IDs for users and items."""
    # Filter data
    print("before filtering", len(score_data))
    score_data = filter_score_data(
        score_data,
        min_scores_per_user=config.min_scores_per_user,
        min_scores_per_item=config.min_scores_per_item,
    )
    user_data = {k: user_data[k] for k in score_data.keys()}
    print("after filtering", len(score_data))

    # Index data
    user_to_id, item_to_id = index_users_and_items(score_data)
    return score_data, user_data, user_to_id, item_to_id


def get_train_and_val_datasets(score_data, user_data, user_to_id, item_to_id):
    # Vectorize
    input_scores, target_scores = vectorize_score_data(
        score_data,
        user_to_id,
        item_to_id,
        sparse=config.use_sparse_score_matrices,
        dtype="float32",
    )
    input_scores = scale_score_matrix(input_scores)
    target_scores = scale_score_matrix(target_scores)

    # Split users between train and test
    users = sorted(score_data.keys())
    num_train_samples = round(config.train_fraction * len(users))

    train_input_scores = input_scores[:num_train_samples]
    train_target_scores = target_scores[:num_train_samples]

    val_input_scores = input_scores[num_train_samples:]
    val_target_scores = target_scores[num_train_samples:]

    from baseline import compute_baseline_metrics
    print(compute_baseline_metrics(train_input_scores, val_target_scores))

    user_features = prepare_user_features(user_data, user_to_id)
    train_user_features = {k: v[num_train_samples:] for k, v in user_features.items()}
    val_user_features = {k: v[:num_train_samples] for k, v in user_features.items()}

    # Preprocess user features
    user_features_preprocessor = make_user_features_preprocessor(
        train_user_features, feature_config=config.user_features_config
    )

    # Make streaming datasets
    train_ds = make_dataset(
        train_input_scores,
        train_target_scores,
        train_user_features,
        user_features_preprocessor,
        batch_size=config.batch_size,
    )
    val_ds = make_dataset(
        val_input_scores,
        val_target_scores,
        val_user_features,
        user_features_preprocessor,
        batch_size=config.batch_size,
    )
    return train_ds, val_ds


def get_full_dataset(score_data, user_data, user_to_id, item_to_id):
    input_scores, target_scores = vectorize_score_data(
        score_data,
        user_to_id,
        item_to_id,
        sparse=config.use_sparse_score_matrices,
        dtype="float32",
    )
    input_scores = scale_score_matrix(input_scores)
    target_scores = scale_score_matrix(target_scores)

    user_features = prepare_user_features(user_data, user_to_id)
    user_features_preprocessor = make_user_features_preprocessor(
        user_features, feature_config=config.user_features_config
    )
    return make_dataset(
        input_scores,
        target_scores,
        user_features,
        user_features_preprocessor,
        batch_size=config.batch_size,
    )
