# Loading Data
from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.linalg import eigh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("Zeller.csv", delimiter=",")
conditions = data["study_condition"].values

X = data.drop(columns=["study_condition"]).select_dtypes(
    include=[np.number]).values


# --- Compute Bray-Curtis distance matrix ---
dist_matrix = squareform(pdist(X, metric="braycurtis"))

# --- PCoA via double centering ---
def pcoa(dist_matrix, k):
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

coords, Lambdas_2D = pcoa(dist_matrix, k=2)
coords, Lambdas_3D = pcoa(dist_matrix, k=3)


# --- 2D Variance explained ---
pos_lambdas_2D = Lambdas_2D[Lambdas_2D > 0]
var_explained_2D = Lambdas_2D[:2] / pos_lambdas_2D.sum() * 100

# --- 3D Variance explained ---
pos_lambdas_3D = Lambdas_3D[Lambdas_3D > 0]
var_explained_3D = Lambdas_3D[:3] / pos_lambdas_3D.sum() * 100

# # --- Plotting in 2D ---
# plt.figure(figsize=(7, 6))
# ax = plt.subplot(111)

# colors = {"CRC": "red", "control": "blue"}

# for c in np.unique(conditions):
#     mask = conditions == c
#     ax.scatter(
#         coords[mask, 0], coords[mask, 1],
#         s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
#         label=c, color=colors.get(c)
#     )

# plt.title("PCoA of Microbiome Data (Bray-Curtis)")
# ax.set_xlabel(f"PC1 ({var_explained_2D[0]:.1f}%)")
# ax.set_ylabel(f"PC2 ({var_explained_2D[1]:.1f}%)")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()



# --- Plotting in 3D ---
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

colors = {"CRC": "red", "control": "blue"}

for o in np.unique(conditions):
    mask = conditions == o
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
        label=o, color=colors.get(o)
    )

plt.title("PCoA of Microbiome Data (Bray-Curtis)")
ax.set_xlabel(f"PC1 ({var_explained_3D[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var_explained_3D[1]:.1f}%)")
ax.set_zlabel(f"PC2 ({var_explained_3D[2]:.1f}%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()