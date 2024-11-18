# Recommender System Template

A flexible recommender system template using Keras that can be customized for various recommendation tasks. The system uses collaborative filtering with neural embeddings and supports both user and item features.

## Shape of the Problem

This template is designed for:
- User-item interaction data (with optional interaction scores/ratings)
- Optional user features (categorical or numerical)
- Optional item features (categorical or numerical)
- Cold-start recommendations using feature-based embeddings
- Implicit or explicit feedback scenarios

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

## Setup

1. Install dependencies using pip:

    pip install -r requirements.txt

2. Configure settings in `config.py`:
   - Set `csv_fpath` to your data file path
   - Adjust `min_interactions_per_user` and `min_interactions_per_item`
   - Configure model parameters like `embedding_dim`
   - Set training parameters like `batch_size` and `learning_rate`

## Training Strategy

The system uses a two-tower neural network architecture:
1. User tower: Combines user ID embeddings with user features
2. Item tower: Combines item ID embeddings with item features
3. Dot product between user and item embeddings produces affinity scores

Training process:
1. Split data into train/validation sets
2. Train with negative sampling
3. Use early stopping to prevent overfitting
4. Save best model weights

## Usage

1. Customize the `get_raw_data()` function in `data.py` to load your data
2. Run exploratory data analysis:

    python main.py exploratory_data_analysis

3. Train and validate the model:

    python main.py train_validation_model

4. Train on full dataset for production:

    python main.py train_production_model

5. Generate recommendations:

    python main.py compute_predictions

Or run all steps sequentially:

    python main.py

## Expected Training Time

Training time depends on:
- Dataset size
- Number of users/items
- Hardware capabilities
- Embedding dimension
- Batch size

On a modern CPU, expect:
- Small datasets (<100K interactions): minutes
- Medium datasets (100K-1M interactions): hours
- Large datasets (>1M interactions): several hours to days

## What You Should Customize

The main point of customization is the `get_raw_data()` function in `data.py`. This function should:

1. Read your data source
2. Return a tuple of:
   - interaction_data: List of dicts with keys "user", "item", and optional "score"
   - user_data: Dict of user features (optional)
   - item_data: Dict of item features (optional)

## Compatibility

- Python 3.10+
- Keras 3.0+
- Works with any Keras backend (TensorFlow, JAX, or PyTorch)
- Supports CPU and GPU training

## Directory Structure

- `config.py`: Configuration settings
- `data.py`: Data loading and preprocessing
- `eda.py`: Exploratory data analysis
- `models.py`: Model architecture
- `train.py`: Training and prediction logic
- `main.py`: Main execution script
