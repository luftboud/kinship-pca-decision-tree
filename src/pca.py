import numpy as np


def compute_pca_embeddings(faces: np.ndarray, max_features_amount: int = 10) -> np.ndarray:
    a_transp_a = faces.T @ faces
    eigenvalues, eigenvectors = np.linalg.eigh(a_transp_a)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    wk = eigenvectors[:, :max_features_amount]

    return wk
