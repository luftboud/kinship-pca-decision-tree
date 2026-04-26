import numpy as np


def jacobi_eigh(matrix, eps=1e-10, max_iter=1000):
    """
    https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    :param matrix:
    :param eps:
    :param max_iter:
    :return:
    """
    matrix = matrix.copy().astype(float)
    n = matrix.shape[0]
    E = np.eye(n)
    e = np.diag(matrix).copy()

    state = n
    ind = np.zeros(n, dtype=int)
    changed = np.ones(n, dtype=bool)


    def maxind(k):
        if k == n - 1:
            return k

        m = k+1
        for i in range(k+2, n):
            if abs(matrix[k, i]) > abs(matrix[k, m]):
                m = i
        return m

    def update(k, t):
        nonlocal state

        y = e[k]
        e[k] = y + t

        if changed[k] and y == e[k]:
            changed[k] = False
            state  -= 1
        elif not changed[k] and y != e[k]:
            changed[k] = True
            state += 1

    def rotate(k, l, i, j, c, s):
        x = matrix[k, l]
        y = matrix[i, j]

        matrix[k, l] = c * x - s * y
        matrix[i, j] = s * x + c * y

    for k in range(n-1):
        ind[k] = maxind(k)
    ind[n-1] = n-1

    iterations = 0

    while state != 0 and iterations < max_iter:
        iterations+=1

        m=0
        for k in range(1, n-1):
            if abs(matrix[k, ind[k]]) > abs(matrix[m, ind[m]]):
                m = k

        k = m
        l = ind[m]
        p = matrix[k, l]

        if abs(p) < eps:
            break

        y = (e[l] - e[k]) / 2.0
        d = abs(y) + np.sqrt(p * p + y * y)

        if d == 0:
            break

        r = np.sqrt(p * p + d * d)
        c = d / r
        s = p / r
        t = (p * p) / d

        if y < 0:
            s = -s
            t = -t

        matrix[k, l] = 0.0
        update(k, -t)
        update(l, t)

        for i in range(k):
            rotate(i, k, i, l, c, s)

        for i in range(k + 1, l):
            rotate(k, i, i, l, c, s)

        for i in range(l + 1, n):
            rotate(k, i, l, i, c, s)

        for i in range(n):
            x = E[i, k]
            y = E[i, l]

            E[i, k] = c * x - s * y
            E[i, l] = s * x + c * y

        for i in range(n - 1):
            ind[i] = maxind(i)

    idx = np.argsort(e)

    eigenvalues = e[idx]
    eigenvectors = E[:, idx]

    return eigenvalues, eigenvectors

def compute_pca_embeddings(faces: np.ndarray, max_features_amount: int = 10) -> np.ndarray:
    n, d = faces.shape
    print(f"PCA: {n} images, {d} features, keeping top {max_features_amount} components")

    if n < d:
        print(f"PCA: Gram matrix {n}x{n}...")
        M = faces @ faces.T
    else:
        print(f"PCA: covariance matrix {d}x{d}...")
        M = faces.T @ faces

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    if n < d:
        k = min(max_features_amount, n)
        wk = faces.T @ eigenvectors[:, :k]
        col_norms = np.sqrt((wk * wk).sum(axis=0, keepdims=True))
        col_norms[col_norms < 1e-12] = 1.0
        wk = wk / col_norms
    else:
        wk = eigenvectors[:, :max_features_amount]

    print(f"PCA: done, projection matrix shape {wk.shape}")
    return wk.astype(np.float32)
