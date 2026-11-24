"""Lab 9: Dimensionality reduction with PCA."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    print("=== Program 09: Principal Component Analysis ===")
    print("Dataset: Iris")
    print(f"Original features: {X.shape[1]}")
    print(f"Reduced features:  {Z.shape[1]}")
    print()
    print("Explained variance ratio:")
    for i, r in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {r:.3f}  ({r * 100:.1f}%)")
    print(f"  Total kept: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    print()
    print("First 8 samples after PCA (PC1, PC2) and class:")
    names = data.target_names
    for row, label in list(zip(Z, data.target))[:8]:
        print(f"  {row[0]:7.3f}  {row[1]:7.3f}   {names[label]}")
    print()
    recon = pca.inverse_transform(Z)
    err = np.mean((X - recon) ** 2)
    print(f"Mean squared reconstruction error: {err:.4f}")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
