from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.linalg import eigh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("microbeCancerGut/Zeller.csv", delimiter=",")

conditions = data["study_condition"].values


X = data.drop(columns=["study_condition"]).select_dtypes(
    include=[np.number]).values

# --- Compute Bray-Curtis distance matrix ---
dist_matrix = squareform(pdist(X, metric="braycurtis"))

# --- PCoA via double centering ---


def pcoa(dist_matrix, k=2):
    n = dist_matrix.shape[0]
    D2 = dist_matrix ** 2

    # Double center
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    # Eigen decomposition
    Lambdas, Q = eigh(B)

    # Sort descending
    idx = np.argsort(Lambdas)[::-1]
    Lambdas, Q = Lambdas[idx], Q[:, idx]

    # Keep only positive eigenvalues
    coords = Q[:, :k] * np.sqrt(np.maximum(Lambdas[:k], 0))
    return coords, Lambdas


coords, Lambdas = pcoa(dist_matrix, k=2)

# --- Variance explained ---
pos_lambdas = Lambdas[Lambdas > 0]
var_explained = Lambdas[:2] / pos_lambdas.sum() * 100

plt.figure(figsize=(7, 6))
ax = plt.subplot(111)

colors = {"CRC": "red", "control": "blue"}

for c in np.unique(conditions):
    mask = conditions == c
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
        label=c, color=colors.get(c)
    )

plt.title("PCoA of Microbiome Data (Bray-Curtis)")
ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
