"""Optional per-element P/Q weighting for the existing baseline losses."""

import math

import torch


def register_element_weights(model, feature_count, q_idx, q_weight):
    weights = None
    if q_weight is not None:
        q_weight = float(q_weight)
        if not math.isfinite(q_weight) or q_weight <= 0:
            raise ValueError("training.pq_element_weight must be finite and positive")
        weights = torch.ones(feature_count)
        weights[list(q_idx)] = q_weight
    model.register_buffer("pq_element_weights", weights, persistent=False)


def weighted_mae(pred, target, mask, weights):
    weighted_mask = mask * weights
    return ((pred - target).abs() * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)


def weighted_rmse(pred, target, mask, weights):
    weighted_mask = mask * weights
    mse = ((pred - target).square() * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)
    # An empty mask or an exact fit has zero loss and zero gradient.
    positive = mse > 0
    root = torch.where(positive, mse, torch.ones_like(mse)).sqrt()
    return torch.where(positive, root, mse)
