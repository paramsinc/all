import data
from config import config
import train
import baseline


if __name__ == "__main__":
    # Load raw data
    print("Loading data...")
    score_data = data.get_score_data()
    user_data = data.get_user_data()

    # Filter users and items with insufficient data
    print("Filtering data...")
    score_data, user_data, user_to_id, item_to_id = data.filter_and_index_data(
        score_data, user_data
    )

    # Use a validation split to find the best hps
    print("Making datasets...")
    train_ds, val_ds = data.get_train_and_val_datasets(
        score_data, user_data, user_to_id, item_to_id
    )

    print("Running hyperparameter search...")
    hp_config = train.get_best_hp_config(train_ds, val_ds)
    print("Best hp config:", hp_config)

    # Train a model on the full dataset with the best hps
    print("Training production model...")
    full_ds = data.get_full_dataset(score_data, user_data, user_to_id, item_to_id)
    model, _ = train.get_model(hp_config, full_ds)

    # Save the model
    model.save(config.prod_model_path)
