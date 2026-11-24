"""Lab 5: Decision tree for a simple classification problem."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=3, stratify=data.target
    )
    tree = DecisionTreeClassifier(max_depth=4, random_state=3)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)

    print("=== Program 05: Decision Tree Classification ===")
    print("Dataset: Wine (3 classes)")
    print(f"Max depth: 4    Train: {len(y_train)}    Test: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, pred))
    print()
    print("Top features used by the tree:")
    order = tree.feature_importances_.argsort()[::-1]
    for i in order[:5]:
        if tree.feature_importances_[i] > 0:
            print(f"  {data.feature_names[i]:22} {tree.feature_importances_[i]:.3f}")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
