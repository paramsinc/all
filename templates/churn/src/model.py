from keras import layers, models, ops
from keras_nlp.layers import TransformerEncoder

from src.config import config
from src.data import event_type_to_int


def create_model() -> models.Model:
    # Input layers
    event_input = layers.Input(
        shape=(config.max_embedding_len,), name="event_input", dtype="int32"
    )
    time_input = layers.Input(
        shape=(config.max_embedding_len,), name="time_input", dtype="float32"
    )

    # Event type embedding
    event_embedding = layers.Embedding(
        input_dim=len(event_type_to_int) + 1,  # Plus one for padding token
        output_dim=config.event_embedding_output_dim,
    )(event_input)

    # Expand time_input dimensions to match event_embedding
    time_features = layers.Reshape((-1, 1))(time_input)

    # Create mask where padding (0) is False, and non-padding is True
    mask = ops.cast(
        ops.not_equal(event_input, 0),
        dtype="float32",
    )  # Shape: (batch_size, sequence_length)
    # Expand mask dimensions to match the embedding dimensions
    mask = ops.expand_dims(mask, axis=-1)  # Shape: (batch_size, sequence_length, 1)

    # Apply mask
    event_embedding = event_embedding * mask  # Broadcasting works here
    time_features = time_features * mask

    # Concatenate embeddings and time differences
    x = layers.Concatenate()([event_embedding, time_features])

    # Squeeze the mask to match expected shape
    padding_mask = ops.squeeze(mask, axis=-1)  # Shape: (batch_size, sequence_length)

    # now run the transformer layers
    for c in config.transformer_configs:
        x = TransformerEncoder(
            intermediate_dim=c["intermediate_dim"],
            num_heads=c["num_heads"],
            dropout=c["dropout"],
        )(x, padding_mask=padding_mask)

    # Global pooling and output layer
    global_pool = layers.GlobalAveragePooling1D()(x)
    output_layer = layers.Dense(1, activation="sigmoid")(global_pool)

    # Define the model
    model = models.Model(inputs=[event_input, time_input], outputs=output_layer)

    return model
