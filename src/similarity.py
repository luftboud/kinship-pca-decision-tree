import numpy as np


def compute_similarity_features(z1, z2):
    z1 = np.asarray(z1, dtype=float).ravel()
    z2 = np.asarray(z2, dtype=float).ravel()
    difference = np.abs(z1 - z2)
    euclidian_distance = np.linalg.norm(z1 - z2)
    manhattan_distance = np.sum(difference)

    denominator = np.linalg.norm(z1) * np.linalg.norm(z2)
    cos_similarity = 0.0 if denominator == 0 else (z1 @ z2) / denominator
    mean = np.mean(difference)
    max_diff = np.max(difference)
    std_diff = np.std(difference)

    outp = np.array([euclidian_distance, manhattan_distance, cos_similarity, mean, max_diff, std_diff])
    return outp


def build_pair_feature_matrix(embeddings, pairs):
    features = [compute_similarity_features(embeddings[i], embeddings[j]) for i, j in pairs]
    return np.array(features, dtype=float)