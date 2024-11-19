import keras

config = keras.utils.Config()

config.data_path = "data/"

config.churn_days_out = 30
config.min_events = 10

config.max_embedding_len = 50

config.test_size = 0.2
config.random_state = 42


config.epochs = 15
config.batch_size = 2048

config.time_diff_scaler_path = "models/min_max_scaler.pkl"
config.prod_model_path = "models/prod_model.keras"

# model config
config.event_embedding_output_dim = 256
config.transformer_configs = [
    {
        "intermediate_dim": 512,
        "num_heads": 4,
        "dropout": 0.2,
    }
]
config.model_dropout_rate = 0.1
