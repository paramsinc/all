import keras


class Encoder(keras.Model):
    """Model that embeds objects via their IDs and optionally features."""

    def __init__(self, num_objects, embedding_dim, features_shape=None, dtype=None):
        index_input = keras.Input(shape=(), name="id")
        embeddings = keras.layers.Embedding(num_objects, embedding_dim)(index_input)
        if features_shape is not None:
            features_input = keras.Input(shape=features_shape, name="features")
            inputs = {"id": index_input, "features": features_input}
            total_features = keras.ops.concatenate(
                [embeddings, features_input], axis=-1
            )
            total_features = keras.layers.Dense(embedding_dim, activation="relu")(
                total_features
            )
            total_features = keras.layers.Dropout(0.3)(total_features)
            outputs = keras.layers.Dense(embedding_dim)(total_features)
        else:
            inputs = {"id": index_input}
            outputs = embeddings
        outputs = keras.layers.UnitNormalization()(outputs)
        super().__init__(inputs=inputs, outputs=outputs)


class EmbeddingModel(keras.Model):
    """Model that trains embeddings of users and items.

    The training is configured so that the dot product
    between a user embedding and an item embedding produces
    a prediction as to the potential interaction score (affinity)
    between the user and the item.

    Call input structure: dict with fields `"user_id"` and `"item_id"`,
    as well as optional fields `"user_features"` and `"item_features"`.

    IDs are integer arrays, which features are dense float arrays.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        user_features_shape=None,
        item_features_shape=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_encoder = Encoder(
            num_users,
            embedding_dim,
            features_shape=user_features_shape,
        )
        self.item_encoder = Encoder(
            num_items,
            embedding_dim,
            features_shape=item_features_shape,
        )
        self.loss_fn = (
            keras.losses.MeanSquaredError()
        )  # keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=False):
        user_input = {"id": inputs["user_id"]}
        if "user_features" in inputs:
            user_input["features"] = inputs["user_features"]
        user_embeddings = self.user_encoder(user_input)

        item_input = {"id": inputs["item_id"]}
        if "item_features" in inputs:
            item_input["features"] = inputs["item_features"]
        item_embeddings = self.item_encoder(item_input)
        return {
            "user_embeddings": user_embeddings,
            "item_embeddings": item_embeddings,
        }

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        affinity = keras.ops.sum(
            keras.ops.multiply(y_pred["user_embeddings"], y_pred["item_embeddings"]),
            axis=1,
        )
        return self.loss_fn(y, affinity, sample_weight)
