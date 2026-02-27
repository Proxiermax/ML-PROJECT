import numpy as np


class MLPScratch:
    """Multi-Layer Perceptron built from scratch with backpropagation.

    Architecture: Input -> Hidden_1 -> Hidden_2 -> Output (sigmoid)
    """

    def __init__(self, hidden_sizes=(64, 32), learning_rate=0.01,
                 n_iterations=500, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.accuracy_history = []

    # ---------- activations ----------
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(a):
        return a * (1.0 - a)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    # ---------- loss ----------
    def _bce_loss(self, y, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # ---------- init ----------
    def _init_params(self, n_input):
        rng = np.random.RandomState(self.random_state)
        layer_sizes = [n_input] + list(self.hidden_sizes) + [1]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialisation
            w = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # ---------- forward ----------
    def _forward(self, X):
        activations = [X]
        z_list = []

        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_list.append(z)

            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)       # output layer
            activations.append(a)

        return activations, z_list

    # ---------- backward ----------
    def _backward(self, activations, z_list, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        n_layers = len(self.weights)

        # output layer gradient
        delta = activations[-1] - y        # dL/dz  for BCE + sigmoid

        dw_list = [None] * n_layers
        db_list = [None] * n_layers

        for i in reversed(range(n_layers)):
            dw_list[i] = (activations[i].T @ delta) / m
            db_list[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_deriv(z_list[i - 1])

        return dw_list, db_list

    # ---------- fit ----------
    def fit(self, X, y):
        self._init_params(X.shape[1])
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(self.n_iterations):
            activations, z_list = self._forward(X)
            y_pred = activations[-1].flatten()

            loss = self._bce_loss(y, y_pred)
            self.loss_history.append(loss)

            acc = np.mean((y_pred >= 0.5).astype(int) == y)
            self.accuracy_history.append(acc)

            dw_list, db_list = self._backward(activations, z_list, y)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw_list[i]
                self.biases[i] -= self.learning_rate * db_list[i]

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}, Loss: {loss:.6f}, Acc: {acc:.4f}")

        return self

    # ---------- predict ----------
    def predict_proba(self, X):
        activations, _ = self._forward(X)
        return activations[-1].flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
