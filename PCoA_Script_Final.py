import numpy as np

# --- Computing the distance matrix D ---
def distance_matrix(X):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D

# --- Computing the centered distance matrix ---
def centering_matrix(X):
    n = X.shape[0]
    H = np.eye(n) - np.ones((n,n)) / n
    D = distance_matrix(X)
    B = -0.5 * H @ (D**2) @ H
    return B


# --- Computing the lower dimensional matrix Z ---
def pcoa(X, k: int):
    B = centering_matrix(X)
    Lambdas, Q = np.linalg.eigh(B)

    idx = np.argsort(Lambdas)[::-1]
    Lambdas = Lambdas[idx]
    Q = Q[:, idx]

    positive_indices = np.where(Lambdas > 0)[0][:k]

    # top k dimensions
    Lambdas_k = Lambdas[:k]
    Q_k = Q[:, :k]

    Z = Q_k * np.sqrt(Lambdas_k)

    return Z, Lambdas_k 