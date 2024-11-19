import logging

import tensorflow as tf

from src.config import config
from src.eval import plot_eval
from src.model import create_model
from src.train import split_data, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMIT = None


def configure_memory() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"Memory configuration failed: {e}")


if __name__ == "__main__":
    configure_memory()

    try:
        logger.info("Preparing data...")
        X_train_event, X_test_event, X_train_time, X_test_time, y_train, y_test = (
            split_data(limit=LIMIT)
        )

        logger.info("Creating model...")
        model = create_model()
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        print(model.summary())

        logger.info("Training model...")
        history = train_model(
            model=model,
            X_train_event=X_train_event,
            X_test_event=X_test_event,
            X_train_time=X_train_time,
            X_test_time=X_test_time,
            y_train=y_train,
            y_test=y_test,
        )

        logger.info("Evaluating model...")
        plot_eval(
            model=model,
            X_test_event=X_test_event,
            X_test_time=X_test_time,
            y_test=y_test,
            history=history,
        )

        logger.info(f"Saving model to {config.prod_model_path}...")
        model.save(config.prod_model_path)
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
