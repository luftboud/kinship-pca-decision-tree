import numpy as np


def mean_centering(face: np.ndarray) -> np.ndarray:
    n_faces, _ = face.shape
    for i in range(n_faces):
        curr_mean = np.mean(face[i])
        face[i] -= curr_mean

    return face


def compute_pca_embeddings(face: np.ndarray, max_features_amount: int = 10) -> np.ndarray:
    face = mean_centering(face)

    a_transp_a = face.T @ face
    eigenvalues, eigenvectors = np.linalg.eigh(a_transp_a)

    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[idx]

    wk = eigenvectors[:, :max_features_amount]
    embeddings = wk.T @ face.T

    return embeddings
