import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

from .config import config
from .data import get_padded_sequences_and_time_diffs_and_labels_from_config


def split_data(
    limit: int | None = None,
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[float]],
    list[list[float]],
    list[int],
    list[int],
]:
    padded_sequences, padded_time_diffs, labels = (
        get_padded_sequences_and_time_diffs_and_labels_from_config(limit=limit)
    )
    return train_test_split(
        padded_sequences,
        padded_time_diffs,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )


def train_model(
    model: keras.Model,
    X_train_event: list[list[int]],
    X_test_event: list[list[int]],
    X_train_time: list[list[float]],
    X_test_time: list[list[float]],
    y_train: list[int],
    y_test: list[int],
) -> keras.callbacks.History:
    history = model.fit(
        {"event_input": X_train_event, "time_input": X_train_time},
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(
            {"event_input": X_test_event, "time_input": X_test_time},
            y_test,
        ),
        callbacks=[TqdmCallback(verbose=1)],
    )
    return history
