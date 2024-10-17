from code.config import config

import keras
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

from .data import labels, padded_sequences, padded_time_diffs

# %%


def split_data() -> tuple[list, list, list, list, list, list]:
    return train_test_split(
        padded_sequences,
        padded_time_diffs,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )


# %%
def train_model(
    model, X_train_event, X_test_event, X_train_time, X_test_time, y_train, y_test
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
