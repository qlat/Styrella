import math

import numpy as np


def clark(a, b):

    a[a == 0] = np.finfo(float).eps
    b[b == 0] = np.finfo(float).eps

    return max(np.sqrt(np.sum(np.power(np.divide(np.fabs(a - b), a + b), 2))), 0)


def sim_cosine(a, b):

    a[a == 0] = np.finfo(float).eps
    b[b == 0] = np.finfo(float).eps

    return np.divide(np.sum(a * b), np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))


def dist_cosine(a, b):

    a[a == 0] = np.finfo(float).eps
    b[b == 0] = np.finfo(float).eps

    num = np.sum(a * b)
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))

    return 1 - (num / denom)


def euclidean(a, b):
    return max(math.sqrt(square_euclidean(a, b)), 0)


def jdivergence(a, b):

    a[a == 0] = np.finfo(float).eps
    b[b == 0] = np.finfo(float).eps

    return np.sum((a - b) * np.log(np.divide(a, b)))


def manhattan(a, b):
    return np.sum(np.fabs(a - b))


def matusita(a, b):
    return max(np.sqrt(np.sum(np.power(np.sqrt(a) - np.sqrt(b), 2))), 0)


def square_euclidean(a, b):
    diff = a - b
    return max(np.sum(diff ** 2), 0)


def tanimoto(a, b):
    return max(manhattan(a, b) / np.sum(np.maximum(a, b)), 0)


def delta(z_scores_a, z_scores_b):
    return np.divide(np.sum(np.fabs(z_scores_a - z_scores_b)), len(z_scores_a))


def eders_delta(z_scores_a, z_scores_b):

    n = len(z_scores_a)
    n_i = np.arange(n) + 1

    return np.divide(np.sum(np.fabs(z_scores_a - z_scores_b) * ((n - n_i + 1) / n)), n)


def delta_cosine(z_scores_a, z_scores_b):

    z_scores_a[z_scores_a == 0] = np.finfo(float).eps
    z_scores_b[z_scores_b == 0] = np.finfo(float).eps

    num = np.sum(z_scores_a * z_scores_b)
    denom = np.sqrt(np.sum(z_scores_a*z_scores_a)) * np.sqrt(np.sum(z_scores_b*z_scores_b))

    return 1 - (num / denom)


def argamon(z_scores_a, z_scores_b):

    return (1 / len(z_scores_a)) * np.sqrt(np.sum(np.power(z_scores_a - z_scores_b, 2)))


delta_measures = {
    'delta': delta,
    'eders': eders_delta,
    'cosine': delta_cosine,
    'argamons': argamon
}


distance_measures = {
    'clark': clark,
    'cosine': dist_cosine,
    'euclidean': euclidean,
    'jdivergence': jdivergence,
    'manhattan': manhattan,
    'matusita': matusita,
    'square_euclidean': square_euclidean,
    'tanimoto': tanimoto
}