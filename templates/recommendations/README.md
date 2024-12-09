# Recommender System Template

A flexible recommender system template using Keras 3 that can be customized for various recommendation tasks.
The system uses collaborative filtering with neural embeddings and supports both user and item features.

## Shape of the Problem

We're considering a fixed set of users, as well as a fixed set of "items" (for instance products, songs, movies...).
We're looking at "interactions" between users and items (for instance, "this user has listening to this song").
An interaction can optionally have a score associated with it (for instance, "this user has rating to this song 4/5").
Ratings are optional: the template will work with only binary interaction data or with ratings all the same.

Based on interaction data, we're computing *embeddings* for users and for items. The dot product between
a user's embedding and an item's embedding produces an *affinity value* between the user and the item, which captures
how likely the user is to want to interact with the item in the future.

Both users and items can optionally have features (such as age, gender, location, etc. for users, or genre, price, etc. for items)
which the model can take into account to improve its predictions. If no such features are available in your case, that's fine:
the template will work with only interaction data.

## Key Inputs and Outputs

### Inputs

1. Interaction data in CSV format with columns:
   - User identifiers
   - Item identifiers 
   - Optional: interaction scores/ratings

2. Optional user features
3. Optional item features

### Outputs

- Trained recommendation model
- Top-N recommendations per user saved to `recommendations.json`
- Optional: EDA visualizations saved to the `figures/` directory

## Usage

1. Install dependencies using pip: `pip install -r requirements.txt`

2. Configure essential settings in `config.py`:
    - Set `raw_data_fpath` to your data file path
    - Adjust `min_interactions_per_user` and `min_interactions_per_item`

3. Optionally, configure non-essential modeling settings in `config.py`:
    - Configure model parameters like `embedding_dim`
    - Set training parameters like `batch_size` and `learning_rate`

4. Edit the function `get_raw_data()` in `data.py` to make sure
it can parse your raw data file and turn it into the expected list of interaction dicts.
This function should read your raw data file and return a tuple of:
    - interaction_data: List of dicts with keys "user", "item", and optional "score"
    - user_data: Dict of user features (optional)
    - item_data: Dict of item features (optional)

5. Run `python main.py` to start training a model. The following actions are available:
    - `python main.py exploratory_data_analysis`: Runs exploratory data analysis over the interaction data and saves figures in the `figures` folder
        (e.g. histograms of interaction counts of user, per item, etc.)
    - `python main.py train_validation_model`: Trains a validation model and display its accuracy on a held-out validation set.
    - `python main.py train_production_model`: Trains the production model on the entire dataset.
    - `python main.py compute_predictions`: Uses the production model to compute recommendations for all users, saved at `recommendations.json`.
        Recommendations for a user do not include any items that the user previously interacted with.

## Directory Structure

- `config.py`: Configuration settings
- `data.py`: Data loading and preprocessing
- `eda.py`: Exploratory data analysis
- `models.py`: Model architecture
- `train.py`: Training and prediction logic
- `main.py`: Main execution script

## Approach

### Model

The model is a two-tower neural network architecture, defined in `model.py`, consisting of:

- User encoder (`Encoder` class): Combines user ID embeddings with user features to produce user embeddings
- Item encoder (`Encoder` class): Combines item ID embeddings with item features to produce item embeddings
- Embedding model (`EmbeddingModel` class): Applies both towers and uses dot product between user and item embeddings produces affinity scores

### Dataset preparation

The data preparation logic is defined in `data.py`. The model is trained on batches that contain *positive samples* (data about an interaction that took place between a user and an item) as well as *negative samples* (data about the lack of interaction between a user and an item).

This happens in the `InteractionDataset` class, which is a Keras `PyDataset`.

### Training strategy

The training code is defined in `train.py`.

- Use early stopping to prevent overfitting (`EarlyStopping` callback)
- Save best model weights continuously during training (`ModelCheckpoint` callback)


## Expected Training Time

Training time depends on:
- Dataset size
- Number of users/items
- Hardware capabilities
- Embedding dimension
- Batch size

On a modern CPU, expect:
- Small datasets (<100K interactions): minutes -- tractable on CPU in hours
- Medium datasets (100K-1M interactions): hours -- requires a GPU
- Large datasets (>1M interactions): several hours to days -- requires a GPU

## Compatibility

- Python 3.10+
- Keras 3.0+
- Works with any Keras backend (TensorFlow, JAX, or PyTorch)
- Supports CPU and GPU training

