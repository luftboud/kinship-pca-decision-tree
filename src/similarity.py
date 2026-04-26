import numpy as np


def compute_similarity_features(z1, z2):
    z1 = np.asarray(z1, dtype=float).ravel()
    z2 = np.asarray(z2, dtype=float).ravel()
    abs_diff = np.abs(z1 - z2)
    prod = z1 * z2

    denom = np.linalg.norm(z1) * np.linalg.norm(z2)
    cos_sim = 0.0 if denom == 0 else float((z1 @ z2) / denom)
    l2 = float(np.linalg.norm(z1 - z2))

    return np.concatenate([
        abs_diff,
        prod,
        np.array([cos_sim, l2], dtype=float)
    ]).astype(np.float32)


def build_pair_feature_matrix(embeddings, pairs):
    features = [compute_similarity_features(embeddings[i], embeddings[j]) for i, j in pairs]
    return np.array(features, dtype=np.float32)