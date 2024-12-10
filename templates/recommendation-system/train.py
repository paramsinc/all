import os
import keras
from keras import ops

from config import config


@keras.saving.register_keras_serializable(package="recsys")
def masked_binary_crossentropy(y_true, y_pred, mask_value=0):
    """Computes the mean crossentropy over known scores only.

    Args:
        y_true: The true score tensor.
        y_pred: The predicted score tensor.

    Returns:
        Scalar tensor, the computed masked error.
    """
    mask = ops.cast(ops.not_equal(y_true, mask_value), dtype=y_pred.dtype)
    raw_error = ops.binary_crossentropy(y_true, y_pred) * mask
    masked_error = ops.sum(raw_error, axis=-1) / (
        ops.sum(mask, axis=-1) + keras.config.epsilon()
    )
    return masked_error


@keras.saving.register_keras_serializable(package="recsys")
def masked_mse(y_true, y_pred, mask_value=0):
    """Computes the mean MSE over known scores only.

    Args:
        y_true: The true score tensor.
        y_pred: The predicted score tensor.

    Returns:
        Scalar tensor, the computed masked error.
    """
    mask = ops.cast(ops.not_equal(y_true, mask_value), dtype=y_pred.dtype)
    squared_diff = ops.square(y_true - y_pred) * mask
    return ops.sum(squared_diff, axis=-1) / (
        ops.sum(mask, axis=-1) + keras.config.epsilon()
    )


@keras.saving.register_keras_serializable(package="recsys")
def masked_mae(y_true, y_pred, mask_value=0):
    """Computes the mean absolute error over known scores only, and unscale it.

    Args:
        y_true: The true score tensor.
        y_pred: The predicted score tensor.

    Returns:
        Scalar tensor, the computed masked error.
    """
    mask = ops.cast(ops.not_equal(y_true, mask_value), dtype=y_pred.dtype)
    raw_error = ops.abs(y_true - y_pred) * mask
    masked_error = ops.sum(raw_error, axis=-1) / (
        ops.sum(mask, axis=-1) + keras.config.epsilon()
    )
    if config.score_scaling_factor is not None:
        return masked_error * config.score_scaling_factor
    return masked_error


def train_model(model, train_ds, val_ds=None, num_epochs=None):
    """Train and evaluate a model.

    Args:
        model: Keras model instance.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        num_epoch: Optional number of epochs to train for.
            If unspecified, we use Early Stopping.
            The best stopping epoch gets returned as the
            "best_epoch" entry in the return dict.

    Returns:
        Dict with keys best_val_error and best_epoch.
    """
    if val_ds is None:
        monitor = "loss"
    else:
        monitor = "val_loss"
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{config.checkpoint_dir}/best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor=monitor,
        )
    ]
    metrics = [
        masked_mae,
    ]
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = masked_binary_crossentropy
    if num_epochs is None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                patience=config.early_stopping_patience,
                monitor=monitor,
                verbose=1,
                restore_best_weights=True,
            )
        )
        num_epochs = config.max_epochs

    # Train the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_ds, epochs=num_epochs, callbacks=callbacks, validation_data=val_ds)

    if val_ds:
        # Evaluate best model
        results = model.evaluate(val_ds, return_dict=True)
    else:
        results = model.evaluate(train_ds, return_dict=True)

    if isinstance(callbacks[-1], keras.callbacks.EarlyStopping):
        num_epochs = callbacks[-1].stopped_epoch

    return {
        "best_val_error": results["masked_mae"],
        "best_epoch": num_epochs,
    }


def get_model(hp_config, train_ds, val_ds=None):
    """Creates, trains and evaluates a model based on a hp config."""
    for x, y in train_ds.take(1):
        num_features = x.shape[-1]
        num_targets = y.shape[-1]

    inputs = keras.Input(shape=(num_features,), name="inputs")

    x = keras.layers.Dense(hp_config.layer_size, activation="relu")(inputs)
    for _ in range(hp_config.num_blocks):
        residual = x
        x = keras.layers.Dense(hp_config.layer_size, activation="relu")(x)
        x = keras.layers.Dense(hp_config.layer_size, activation="relu")(x)
        x = keras.layers.Dropout(hp_config.dropout_rate)(x)
        x = x + residual

    outputs = keras.layers.Dense(num_targets, activation="sigmoid", name="outputs")(x)
    model = keras.Model(inputs, outputs, name="score_prediction_model")
    model.summary()

    results = train_model(
        model, train_ds, val_ds=val_ds, num_epochs=hp_config.get("best_epoch", None)
    )
    return model, results


def get_best_hp_config(train_ds, val_ds):
    """Implements elementary hyperparameter search.

    For anything more sophisticated, you should use KerasTuner.
    """
    all_results = []
    for num_blocks in (1, 2):
        for layer_size in (512, 1024, 2048):
            hp_config = keras.utils.Config(
                num_blocks=num_blocks,
                layer_size=layer_size,
                dropout_rate=0.3,
            )
            print("Trying config: ", hp_config)
            _, results = get_model(hp_config, train_ds, val_ds=val_ds)
            results["hp_config"] = hp_config
            all_results.append(results)
    all_results.sort(key=lambda x: x["best_val_error"])
    best_hp_config = all_results[0]["hp_config"]
    best_hp_config["best_epoch"] = all_results[0]["best_epoch"]
    return best_hp_config
