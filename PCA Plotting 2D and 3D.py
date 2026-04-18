# Loading Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("Zeller.csv", delimiter=",")
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

coords, Lambdas_2D = pca_from_cov(X, k=2)
coords, Lambdas_3D = pca_from_cov(X, k=3)



# --- 2D Variance explained ---
pos_lambdas_2D = Lambdas_2D[Lambdas_2D > 0]
var_explained_2D = Lambdas_2D[:2] / pos_lambdas_2D.sum() * 100

# --- 3D Variance explained ---
pos_lambdas_3D = Lambdas_3D[Lambdas_3D > 0]
var_explained_3D = Lambdas_3D[:3] / pos_lambdas_3D.sum() * 100

Z2, Lambdas_2D = pca_from_cov(X, 2)
Z3, Lambdas_3D = pca_from_cov(X, 3)


# # --- Plotting in 2D ---
# plt.figure(figsize=(6, 5))
# ax = plt.subplot(111)

# colors = {"CRC": "red", "control": "blue"}
# mask_outlier = np.abs(Z2[:, 0]) < 10

# for c in np.unique(conditions):
#     mask = (conditions == c) & mask_outlier
#     ax.scatter(
#         Z2[mask, 0], Z2[mask, 1], s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
#         label=c, color=colors.get(c)
#     )

# ax.set_title("PCA of Microbiome Data - 2D projection", fontsize=13, weight="bold")
# ax.set_xlabel(f"PC1 ({var_explained_2D[0]:.1f}%)")
# ax.set_ylabel(f"PC2 ({var_explained_2D[1]:.1f}%)")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()


# --- Plotting in 3D ---
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

colors = {"CRC": "red", "control": "blue"}
mask_outlier = np.abs(Z3[:, 0]) < 10

for o in np.unique(conditions):
    mask = (conditions == o) & mask_outlier
    ax.scatter(
        Z3[mask, 0], Z3[mask, 1], Z3[mask, 2], s=70, edgecolors="white", linewidth=1.0, alpha=0.85,
        label=o, color=colors.get(o)
    )

ax.set_title("PCA of Microbiome Data - 3D projection", fontsize=13, weight="bold")
ax.set_xlabel(f"PC1 ({var_explained_3D[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var_explained_3D[1]:.1f}%)")
ax.set_zlabel(f"PC3 ({var_explained_3D[2]:.1f}%)")
ax.legend()
plt.tight_layout()
plt.show()