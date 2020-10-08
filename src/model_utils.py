"""Shared model utils."""
import logging
import numpy as np
import torch
from tqdm import tqdm

def get_distance(rep_a, rep_b, distance_metric):
    # rep_a, rep_b are of size (B, d) each
    B, d = rep_a.shape
    if distance_metric == 'l2':
        distance = torch.norm(rep_a - rep_b, dim=1)  # B,
    elif distance_metric == 'cosine':
        distance = torch.cosine_similarity(rep_a, rep_b)  # B,
    elif distance_metric == 'dot':
        distance = (rep_a * rep_b).sum(-1)  # B,
    else:
        raise ValueError(distance_metric)
    return distance


def fit_logistic_helper(x_features, y_labels, T=10000, lr_init=1e0, return_normalized=False,
                        return_loss=False):
    # Important to make it mean-centered, unit variance, for conditioning
    mu = np.mean(x_features)
    std = np.std(x_features)
    logging.info('fit_logistic(): mean={}, stdev={}'.format(mu, std))
    x_features = (np.array(x_features) - mu) / std
    y_labels = 2 * np.array(y_labels) - 1  # +1 / -1 labels

    # Train logistic regression with GD
    w = np.zeros(1)
    b = np.zeros(1)
    for t in tqdm(range(T), desc='Training in fit_logistic()'):
        lr = lr_init
        margins = y_labels * (w * x_features + b)
        losses = np.log(1 + np.exp(-margins))
        loss = np.mean(losses)
        dz = (-y_labels) * np.exp(-margins) / (1 + np.exp(-margins))
        dw = np.mean(dz * x_features)
        db = np.mean(dz)
        w = w - lr * dw
        b = b - lr * db
        acc = np.mean(margins > 0)
    logging.info('fit_logistic(): loss={}, acc={}, w={}, b={}'.format(loss, acc, w, b))
    # w * (x - mu) / std + b = (w/std) * x + (b - w * mu / std)
    real_w = float(w / std)
    real_b = float(b - w * mu / std)
    logging.info('fit_logistic(): real_w={}, real_b={}, boundary={}'.format(real_w, real_b, -real_b / real_w))
    if return_normalized:
        real_w = w
        real_b = b
    if return_loss:
        return real_w, real_b, loss
    return real_w, real_b
