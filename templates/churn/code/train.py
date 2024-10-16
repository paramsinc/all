from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

from .data import labels, padded_sequences, padded_time_diffs
from .model import model

# %%

# Split the data
X_train_event, X_test_event, X_train_time, X_test_time, y_train, y_test = (
    train_test_split(
        padded_sequences,
        padded_time_diffs,
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
)

# %%

# print(X_train_event[0], X_train_time[0])
# Use a subset of data

# subset_size = 1000  # adjust as needed
# X_train_event_subset = X_train_event[:subset_size]
# X_train_time_subset = X_train_time[:subset_size]
# y_train_subset = y_train[:subset_size]

# TODO get from config

EPOCHS = 10
BATCH_SIZE = 1024

history = model.fit(
    {"event_input": X_train_event, "time_input": X_train_time},
    y_train,
    epochs=EPOCHS,
    # batch_size=32,
    # batch_size=512,
    batch_size=BATCH_SIZE,
    # batch_size=2048,
    # batch_size=4096,
    validation_data=({"event_input": X_test_event, "time_input": X_test_time}, y_test),
    # validation_split=0.2,
    callbacks=[TqdmCallback(verbose=1)],
)
