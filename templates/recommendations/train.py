import keras
import json
import math
import heapq
import numpy as np
import os

import data
import models
from config import config


def train_validation_model(
    interaction_data, user_index, item_index, user_data, item_data
):
    print("Making train & val datasets...")
    train_ds, val_ds = data.get_train_and_val_datasets(
        interaction_data, user_index, item_index, user_data, item_data
    )

    for example_batch, _ in train_ds:
        break

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.checkpoint_dir, "val_model.weights.h5"),
            monitor="loss",
            save_weights_only=True,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    ]

    for example_batch, _ in train_ds:
        break
    if "user_features" in example_batch:
        user_features_shape = example_batch["user_features"].shape[1:]
    else:
        user_features_shape = None
    if "item_features" in example_batch:
        item_features_shape = example_batch["item_features"].shape[1:]
    else:
        item_features_shape = None

    model = models.EmbeddingModel(
        num_users=len(user_index),
        num_items=len(item_index),
        embedding_dim=config.embedding_dim,
        user_features_shape=user_features_shape,
        item_features_shape=item_features_shape,
    )
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate))
    model.fit(
        train_ds, validation_data=val_ds, epochs=config.max_epochs, callbacks=callbacks
    )


def train_production_model(
    interaction_data, user_index, item_index, user_data, item_data
):
    print("Making full dataset...")
    train_ds = data.get_full_dataset(
        interaction_data, user_index, item_index, user_data, item_data
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(config.prod_model_path, save_weights_only=True),
    ]

    for example_batch, _ in train_ds:
        break
    if "user_features" in example_batch:
        user_features_shape = example_batch["user_features"].shape[1:]
    else:
        user_features_shape = None
    if "item_features" in example_batch:
        item_features_shape = example_batch["item_features"].shape[1:]
    else:
        item_features_shape = None

    model = models.EmbeddingModel(
        num_users=len(user_index),
        num_items=len(item_index),
        embedding_dim=config.embedding_dim,
        user_features_shape=user_features_shape,
        item_features_shape=item_features_shape,
    )
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate))
    model.fit(train_ds, epochs=config.max_epochs, callbacks=callbacks)


def compute_predictions(interaction_data, user_index, item_index, user_data, item_data):
    user_features, item_features = data.get_preprocessed_features(
        user_data, item_data, user_index, item_index
    )

    model = models.EmbeddingModel(
        num_users=len(user_index),
        num_items=len(item_index),
        embedding_dim=config.embedding_dim,
        user_features_shape=(
            next(iter(user_features.values())).shape if user_features else None
        ),
        item_features_shape=(
            next(iter(item_features.values())).shape if item_features else None
        ),
    )
    model.load_weights(config.prod_model_path)

    user_data = {"id": np.arange(0, len(user_index), dtype="int32")}
    item_data = {"id": np.arange(0, len(item_index), dtype="int32")}
    reverse_user_index = {v: k for k, v in user_index.items()}
    reverse_item_index = {v: k for k, v in item_index.items()}
    if user_features:
        user_data["features"] = np.stack(
            [user_features[reverse_user_index[i]] for i in range(0, len(user_index))],
            axis=0,
        )
    if item_features:
        item_data["features"] = np.stack(
            [item_features[reverse_item_index[i]] for i in range(0, len(item_index))],
            axis=0,
        )

    user_embeddings = model.user_encoder.predict(
        user_data, batch_size=config.batch_size
    )
    item_embeddings = model.item_encoder.predict(
        item_data, batch_size=config.batch_size
    )

    items = [reverse_item_index[i] for i in range(len(item_index))]
    items_per_user = data.get_items_per_user(interaction_data)
    recommendations = {}

    num_recs = 50  # Number of recommendations per user (excludes known items)
    num_batches = math.ceil(user_embeddings.shape[0] / config.batch_size)
    for i in range(num_batches):
        print(i, "/", num_batches)
        affinity = keras.ops.dot(
            user_embeddings[i * config.batch_size : (i + 1) * config.batch_size],
            keras.ops.transpose(item_embeddings),
        )
        affinity = keras.ops.convert_to_numpy(affinity)
        for j in range(affinity.shape[0]):
            user = reverse_user_index[i * config.batch_size + j]
            known_items = items_per_user[user]
            scores = list(zip(items, affinity[j]))
            scores = heapq.nlargest(
                len(known_items) + num_recs, scores, key=lambda x: x[1]
            )
            selected_scores = [
                (s[0], float(s[1])) for s in scores if s[0] not in known_items
            ]
            selected_scores = selected_scores[:num_recs]
            recommendations[user] = selected_scores

    with open(config.recommendations_json_fpath, "w") as json_file:
        json.dump(recommendations, json_file)
