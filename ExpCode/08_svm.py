"""Lab 8: Support vector machine for binary classification."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=4, stratify=data.target
    )
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("=== Program 08: Support Vector Machine ===")
    print("Dataset: Breast Cancer (binary: malignant / benign)")
    print("Kernel: RBF    C = 1.0")
    print(f"Train: {len(y_train)}    Test: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print()
    print(classification_report(y_test, pred, target_names=data.target_names))
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
