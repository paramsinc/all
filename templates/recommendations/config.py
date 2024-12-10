from pathlib import Path

import keras
from keras.utils import FeatureSpace as fs

config = keras.utils.Config()

# Path to file of scores/rating data
parent = Path(__file__).parent

# update these file paths to point to your data
config.raw_interaction_data_fpath = str(parent / "_data/dummy_data/user_ratings.csv")
config.raw_user_data_fpath = str(parent / "_data/dummy_data/users.csv")
config.raw_item_data_fpath = str(parent / "_data/dummy_data/games.csv")

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
config.user_features_config = None

# Feature config
# update these configs according to your data
# set config.user_features_config = None
# and config.item_features_config = None if not features are available
config.user_features_config = {
    "age": fs.float_normalized(name="age"),
    "gender": fs.string_categorical(
        name="gender", num_oov_indices=0, output_mode="one_hot"
    ),
}
config.item_features_config = {
    "category": fs.string_categorical(
        name="category", num_oov_indices=0, output_mode="one_hot"
    ),
    "difficulty": fs.string_categorical(
        name="difficulty", num_oov_indices=0, output_mode="one_hot"
    ),
}

# Training config
config.batch_size = 512
config.learning_rate = 1e-3
config.max_epochs = 100
config.early_stopping_patience = 4
config.pydataset_use_multiprocessing = True
config.pydataset_workers = 2

# Exploratory data analysis config
config.eda_figures_dpi = 200
config.eda_figures_dir = "figures"
