import typing as T

import tensorflow as tf
from keras import layers, models, ops
from keras_nlp.layers import TransformerEncoder

from src.config import config
from src.data import event_type_to_int


class EventTransformer(models.Model):
    def __init__(
        self,
        max_sequence_length: int,
        num_event_types: int,
        event_embedding_dim: int,
        transformer_configs: list[dict[str, T.Any]],
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.event_embedding = layers.Embedding(
            input_dim=num_event_types + 1,
            output_dim=event_embedding_dim,
            mask_zero=True,
            name="event_embedding",
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_sequence_length,
            output_dim=event_embedding_dim,
            name="position_embedding",
        )

        self.time_projection = layers.Dense(
            event_embedding_dim, activation="relu", name="time_projection"
        )

        self.dropout = layers.Dropout(dropout_rate)

        self.transformer_blocks = [
            TransformerEncoder(
                intermediate_dim=config["intermediate_dim"],
                num_heads=config["num_heads"],
                dropout=config["dropout"],
            )
            for config in transformer_configs
        ]

        self.pooling = layers.GlobalAveragePooling1D()
        self.final_dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(
        self,
        inputs: dict[str, tf.Tensor] | tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        if isinstance(inputs, dict):
            event_input = inputs["event_input"]
            time_input = inputs["time_input"]
        else:
            event_input, time_input = inputs

        positions = ops.arange(0, ops.shape(event_input)[1])
        positions = ops.cast(positions, "int32")

        event_embedding = self.event_embedding(event_input)
        position_embedding = self.position_embedding(positions)
        time_features = self.time_projection(ops.expand_dims(time_input, -1))

        x = event_embedding + position_embedding + time_features
        x = self.dropout(x, training=training)

        mask = self.event_embedding.compute_mask(event_input)

        for transformer in self.transformer_blocks:
            x = transformer(x, padding_mask=mask)

        x = self.pooling(x)
        x = self.final_dropout(x, training=training)
        return self.output_layer(x)


def create_model() -> models.Model:
    return EventTransformer(
        max_sequence_length=config.max_embedding_len,
        num_event_types=len(event_type_to_int),
        event_embedding_dim=config.event_embedding_output_dim,
        transformer_configs=config.transformer_configs,
        dropout_rate=0.1,
    )
