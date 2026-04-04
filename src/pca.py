import numpy as np


def compute_pca_embeddings(face: np.ndarray, max_features_amount: int = 10) -> np.ndarray:
    a_transp_a = face.T @ face
    eigenvalues, eigenvectors = np.linalg.eigh(a_transp_a)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[idx]

    wk = eigenvectors[:, :max_features_amount]
    embeddings = wk.T @ face.T

    return embeddings
