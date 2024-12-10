import numpy as np
from config import config


def compute_baseline_metrics(inputs_matrix, targets_matrix):
    inputs_matrix = inputs_matrix.toarray()
    targets_matrix = targets_matrix.toarray()
    mean_scores_per_item = np.mean(
        inputs_matrix, axis=0, where=inputs_matrix != 0
    ).flatten()
    total_loss = np.subtract(targets_matrix, mean_scores_per_item)
    total_mse = total_loss**2
    total_mse = np.mean(total_loss**2, where=targets_matrix != 0)
    total_mae = np.mean(np.abs(total_loss), where=targets_matrix != 0)
    if config.score_scaling_factor is not None:
        total_mae *= config.score_scaling_factor
    return total_mse, total_mae
