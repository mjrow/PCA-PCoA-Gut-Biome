import numpy as np

# --- Center Data ---
def center_columns(X):
    return X-X.mean(axis=0)

#--- Compute Covariance Matrix
def covariance_matrix(Xc):
    n = Xc.shape[0]
    return (Xc.conj().T @ Xc) / (n - 1)

# --- Returning sorted Eigenvalues and their respective Eigenvectors ---
def eig_sorted_symmetric(C):
    Lambdas, Q = np.linalg.eigh(C)

    idx = np.argsort(Lambdas)[::-1]
    Lambdas = Lambdas[idx]
    Q = Q[:, idx]

    return Lambdas, Q

# --- Computing lower dimensional Matrix Z from covariance matrix ---
def pca_from_cov(X, k: int):
    Xc = center_columns(X)
    C = covariance_matrix(Xc)

    Lambdas, Q = eig_sorted_symmetric(C)

    P = Q[:, :k]
    Z = Xc @ P
    return Z, Lambdas[:k]