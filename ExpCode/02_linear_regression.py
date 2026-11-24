"""Lab 2: Linear regression for a continuous target."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    rng = np.random.default_rng(7)
    # Hours studied -> exam score
    hours = rng.uniform(1, 10, size=(80, 1))
    score = 35 + 6.2 * hours.ravel() + rng.normal(0, 3, size=80)

    X_train, X_test, y_train, y_test = train_test_split(
        hours, score, test_size=0.25, random_state=7
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("=== Program 02: Linear Regression ===")
    print("Task: predict exam score (continuous) from hours studied")
    print(f"Train samples: {len(y_train)}   Test samples: {len(y_test)}")
    print(f"Learned equation: score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * hours")
    print(f"Test MSE: {mean_squared_error(y_test, pred):.2f}")
    print(f"Test R2:  {r2_score(y_test, pred):.3f}")
    print()
    print("Sample predictions:")
    print(f"  {'Hours':>8} {'Actual':>10} {'Predicted':>12}")
    for h, a, p in list(zip(X_test.ravel(), y_test, pred))[:8]:
        print(f"  {h:8.2f} {a:10.2f} {p:12.2f}")
    print()
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
