import numpy as np


class MLPRegressionScratch:
    def __init__(self, input_dim, hidden1=256, hidden2=128, hidden3=64, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, seed=42):
        np.random.seed(seed)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, hidden3) * np.sqrt(2 / hidden2)
        self.b3 = np.zeros((1, hidden3))

        self.W4 = np.random.randn(hidden3, 1) * np.sqrt(2 / hidden3)
        self.b4 = np.zeros((1, 1))

        self.m, self.v = {}, {}
        for param in ["W1","b1","W2","b2","W3","b3","W4","b4"]:
            self.m[param] = 0
            self.v[param] = 0

        self.loss_history = []
        self.val_history = []

        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def leaky_relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def standardize_fit(self, X, y):
        self.X_mean = X.mean(axis=0, keepdims=True)
        self.X_std = X.std(axis=0, keepdims=True) + 1e-8

        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8

    def standardize_transform(self, X, y=None):
        X = (X - self.X_mean) / self.X_std
        if y is not None:
            y = (y - self.y_mean) / self.y_std
            return X, y
        return X

    def inverse_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mean

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.leaky_relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.leaky_relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.leaky_relu(self.Z3)

        self.Z4 = self.A3 @ self.W4 + self.b4
        return self.Z4

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def adam_update(self, param, grad):
        self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
        self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param] / (1 - self.beta2 ** self.t)

        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def backward(self, X, y_true, y_pred):

        self.t += 1  
        m = X.shape[0]

        dZ4 = 2 * (y_pred - y_true) / m
        dW4 = self.A3.T @ dZ4
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.leaky_relu_derivative(self.Z3)
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.leaky_relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        clip = 5
        for grad in [dW1, dW2, dW3, dW4]:
            np.clip(grad, -clip, clip, out=grad)

        self.W4 -= self.adam_update("W4", dW4)
        self.b4 -= self.adam_update("b4", db4)

        self.W3 -= self.adam_update("W3", dW3)
        self.b3 -= self.adam_update("b3", db3)

        self.W2 -= self.adam_update("W2", dW2)
        self.b2 -= self.adam_update("b2", db2)

        self.W1 -= self.adam_update("W1", dW1)
        self.b1 -= self.adam_update("b1", db1)

    def fit(self, X, y, X_val=None, y_val=None, epochs=2000, batch_size=64, early_stopping_rounds=200):

        y = y.reshape(-1, 1)

        self.standardize_fit(X, y)
        X, y = self.standardize_transform(X, y)

        if X_val is not None:
            y_val = y_val.reshape(-1, 1)
            X_val, y_val = self.standardize_transform(X_val, y_val)

        n = X.shape[0]
        best_loss = float("inf")
        patience = 0
        best_weights = None

        for _ in range(epochs):

            indices = np.random.permutation(n)
            X = X[indices]
            y = y[indices]

            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            train_pred = self.forward(X)
            train_loss = self.compute_loss(train_pred, y)
            self.loss_history.append(train_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                self.val_history.append(val_loss)
            else:
                val_loss = train_loss

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), 
                                self.W3.copy(), self.b3.copy(), self.W4.copy(), self.b4.copy())
            else:
                patience += 1

            if patience > early_stopping_rounds:
                break

        if best_weights is not None:
            (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4) = best_weights

    def predict(self, X):
        X = self.standardize_transform(X)
        y_scaled = self.forward(X)
        return self.inverse_y(y_scaled)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot