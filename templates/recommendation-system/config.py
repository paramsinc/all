import keras
from keras.utils import FeatureSpace as fs

config = keras.utils.Config()

# Path to parquet file of scores/rating data
config.score_data_parquet_fpath = "data/affinity_11092024_export.parquet"
config.prod_model_path = "models/prod_model.keras"
config.checkpoint_dir = "models/tmp"

# Minimum number of scores/ratings per user to keep the user in the data
config.min_scores_per_user = 2

# Minimum number of scores/ratings per items to keep the item in the data
config.min_scores_per_item = 50

# Fraction of data to use for training (remainder is for eval)
config.train_fraction = 0.8
# Fraction of scores per user to use as targets
config.target_score_fraction = 0.3

# Training config
config.batch_size = 64
config.learning_rate = 1e-3
config.max_epochs = 100
config.early_stopping_patience = 4

# Whether to use sparse or dense matrices for handling score data
config.use_sparse_score_matrices = True
config.score_scaling_factor = 5.05

config.user_features_config = {
    "gender": fs.string_categorical(
        name="gender", num_oov_indices=0, output_mode="one_hot"
    ),
    "age": fs.float_normalized(name="age"),
}

# EDA config
config.eda_figures_dpi = 200
config.eda_figures_dir_before_filtering = "figures/before_filtering"
config.eda_figures_dir_after_filtering = "figures/after_filtering"
