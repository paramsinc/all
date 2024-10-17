import keras

config = keras.utils.Config()

config.data_path = ""

config.churn_days_out = 30
config.min_events = 20

config.max_embedding_len = 150

config.test_size = 0.2
config.random_state = 42

config.transformer_layers = 3

config.epochs = 10
config.batch_size = 1024

config.prod_model_path = "models/prod_model.keras"
config.prod_model_gcs_path = "models/v1/model.keras"
