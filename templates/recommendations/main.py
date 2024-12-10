import sys

import data
import train
import eda

if __name__ == "__main__":
    action = "compute_predictions"
    if len(sys.argv) > 2:
        raise ValueError(f"Expected exactly one argument, received: {sys.argv[1:]}")
    if len(sys.argv) == 2:
        action = sys.argv[1]
    else:
        action = None
    valid_actions = {
        "exploratory_data_analysis",
        "train_validation_model",
        "train_production_model",
        "compute_predictions",
        None,
    }
    if action not in valid_actions:
        raise ValueError(
            f"Expected argument to be one of {valid_actions}, received: {action}"
        )

    print("Loading raw data...")
    interaction_data, user_data, item_data = data.get_raw_data()
    print("Filtering data...")
    interaction_data, user_index, item_index, user_data, item_data = (
        data.filter_and_index_data(interaction_data, user_data, item_data)
    )
    print("Num interactions:", len(interaction_data))
    print("Num users:", len(user_index))
    print("Num items:", len(item_index))

    if action == "exploratory_data_analysis":
        eda.display_all(interaction_data)

    if action in {None, "train_validation_model"}:
        train.train_validation_model(
            interaction_data, user_index, item_index, user_data, item_data
        )
    if action in {None, "train_production_model"}:
        train.train_production_model(
            interaction_data, user_index, item_index, user_data, item_data
        )
    if action in {None, "compute_predictions"}:
        train.compute_predictions(
            interaction_data, user_index, item_index, user_data, item_data
        )
