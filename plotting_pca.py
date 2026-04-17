# Loading Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("microbeCancerGut/Zeller.csv", delimiter=",")

conditions = data["study_condition"].values

X = data.drop(columns=["study_condition"]).select_dtypes(
    include=[np.number]).values
# --- Center Data ---


def center_columns(X):
    return X-X.mean(axis=0)

# --- Compute Covariance Matrix


def covariance_matrix(Xc):
    n = Xc.shape[0]
    return (Xc.T @ Xc) / (n - 1)

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


Z, Lambdas = pca_from_cov(X, 2)


plt.figure(figsize=(6, 5))
ax = plt.subplot(111)

colors = {"CRC": "red", "control": "blue"}

mask_outlier = np.abs(Z[:, 0]) < 10

for c in np.unique(conditions):
    mask = (conditions == c) & mask_outlier
    ax.scatter(
        Z[mask, 0], Z[mask, 1], s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
        label=c, color=colors.get(c)
    )

plt.title("PCA of Microbiome Data")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
