"""Lab 4: Small neural network for image classification (8x8 digit images).
Uses a 2-layer network trained with backpropagation in NumPy so it runs
without a GPU. Same task as a tiny PyTorch/TensorFlow classifier.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def main():
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1).astype(np.float64)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rng = np.random.default_rng(1)
    n_in, n_h, n_out = 64, 32, 10
    W1 = rng.normal(0, 0.1, size=(n_in, n_h))
    b1 = np.zeros(n_h)
    W2 = rng.normal(0, 0.1, size=(n_h, n_out))
    b2 = np.zeros(n_out)
    Y = np.eye(n_out)[y_train]
    lr = 0.2

    print("=== Program 04: Neural Network Image Classification ===")
    print("Images: sklearn digits, 8x8 pixels, 10 classes (0-9)")
    print(f"Architecture: 64 -> 32 (ReLU) -> 10 (softmax)")
    print(f"Train images: {len(y_train)}   Test images: {len(y_test)}")
    print()

    for epoch in range(1, 41):
        h = np.maximum(0, X_train @ W1 + b1)
        probs = softmax(h @ W2 + b2)
        # backprop
        d2 = (probs - Y) / len(Y)
        dW2 = h.T @ d2
        db2 = d2.sum(axis=0)
        dh = d2 @ W2.T
        dh[h <= 0] = 0
        dW1 = X_train.T @ dh
        db1 = dh.sum(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        if epoch % 10 == 0:
            acc = (probs.argmax(1) == y_train).mean()
            print(f"  Epoch {epoch:2d}   train accuracy {acc:.3f}")

    h_test = np.maximum(0, X_test @ W1 + b1)
    pred = softmax(h_test @ W2 + b2).argmax(1)
    print()
    print(f"Test accuracy: {(pred == y_test).mean():.3f}")
    print("First 12 test predictions vs actual:")
    print("  pred  ", pred[:12].tolist())
    print("  actual", y_test[:12].tolist())
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
