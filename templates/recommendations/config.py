import keras
from keras.utils import FeatureSpace as fs

config = keras.utils.Config()

# Path to file of scores/rating data
config.csv_fpath = "/Users/francoischollet/Downloads/banq.csv" # "FILL/ME.csv"  # 

config.prod_model_path = "models/prod_model.weights.h5"
config.checkpoint_dir = "models/tmp"
config.recommendations_json_fpath = "recommendations.json"

# Minimum number of scores/ratings per user to keep the user in the data
config.min_interactions_per_user = 2
# Minimum number of scores/ratings per items to keep the item in the data
config.min_interactions_per_item = 50

# Fraction of data to use for training (remainder is for eval)
config.per_user_val_fraction = 0.3
config.score_scaling_factor = None
config.negative_sampling_fraction = 0.5

# Model config
config.embedding_dim = 1024
config.user_features_config = {
    "age": fs.string_categorical(name="age", num_oov_indices=0, output_mode="one_hot"),
}
config.item_features_config = {
    "category": fs.integer_categorical(
        name="category", num_oov_indices=0, output_mode="one_hot"
    ),
}

# Training config
config.batch_size = 512
config.learning_rate = 1e-3
config.max_epochs = 100
config.early_stopping_patience = 4
config.pydataset_use_multiprocessing = True
config.pydataset_workers = 2

# EDA config
config.eda_figures_dpi = 200
config.eda_figures_dir = "figures"

