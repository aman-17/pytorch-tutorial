import numpy as np

np.random.seed(42)
n, d = 100, 3   # 100 samples, 3 features
X = np.random.randn(n, d)
true_w = np.array([2.0, -3.0, 1.5])
y = X @ true_w + np.random.randn(n) * 0.5  # add noise
# Add bias (intercept)
X_bias = np.hstack([np.ones((n, 1)), X])   # shape: (n, d+1)
w_closed = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
print("Closed-form weights:", w_closed)

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    for epoch in range(epochs):
        y_pred = X @ w
        grad = (1/n) * (X.T @ (y_pred - y))  # gradient of MSE
        w -= lr * grad
    return w

w_gd = gradient_descent(X_bias, y, lr=0.05, epochs=5000)
print("Gradient descent weights:", w_gd)