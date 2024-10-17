from code.config import config

import tensorflow as tf
from keras import layers, models

from .data import event_type_to_int, max_sequence_length


def transformer_encoder(
    inputs: tf.Tensor, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0
) -> tf.Tensor:
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward network
    ffn_output = layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)


def create_model() -> models.Model:
    # Input layers
    event_input = layers.Input(
        shape=(max_sequence_length,), name="event_input", dtype=tf.int32
    )
    time_input = layers.Input(
        shape=(max_sequence_length,), name="time_input", dtype=tf.float32
    )

    # Event type embedding
    event_embedding = layers.Embedding(
        input_dim=len(event_type_to_int) + 1,  # Plus one for padding token
        output_dim=256,
    )(event_input)

    # Expand time_input dimensions to match event_embedding
    time_features = layers.Reshape((-1, 1))(time_input)

    # Create mask where padding (0) is False, and non-padding is True
    mask = tf.cast(
        tf.not_equal(event_input, 0),
        dtype=tf.float32,
    )  # Shape: (batch_size, sequence_length)
    # Expand mask dimensions to match the embedding dimensions
    mask = tf.expand_dims(mask, axis=-1)  # Shape: (batch_size, sequence_length, 1)

    # Apply mask
    event_embedding = event_embedding * mask  # Broadcasting works here
    time_features = time_features * mask

    # Concatenate embeddings and time differences
    x = layers.Concatenate()([event_embedding, time_features])

    # Transformer-like Encoder layers
    for _ in range(config.transformer_layers):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)

    # Global pooling and output layer
    global_pool = layers.GlobalAveragePooling1D()(x)
    output_layer = layers.Dense(1, activation="sigmoid")(global_pool)

    # Define the model
    model = models.Model(inputs=[event_input, time_input], outputs=output_layer)

    return model
