"""Lab 3: k-nearest neighbors classification."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
    )
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("=== Program 03: K-Nearest Neighbors ===")
    print("Dataset: Iris (3 classes)")
    print(f"k = 5    Train: {len(y_train)}    Test: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print()
    print(classification_report(y_test, pred, target_names=data.target_names))
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    label = data.target_names[model.predict(sample)[0]]
    print(f"New flower [5.1, 3.5, 1.4, 0.2] classified as: {label}")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
