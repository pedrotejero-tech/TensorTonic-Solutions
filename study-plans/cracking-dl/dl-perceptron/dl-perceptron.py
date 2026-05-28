import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    # Initialization
    X = np.asarray(X)
    y = np.asarray(y)
    W = np.zeros(len(X[0]))
    b = 0.0

    for epoch in range(epochs):
      for x, y_true in zip(X, y):
          # Forward pass
          z = np.dot(W, x) + b
          y_pred = 1 if z >= 0 else 0

          # Loss
          loss = y_true - y_pred

          # Backward pass
          W += lr * loss * x
          b += lr * loss

    return (W.tolist(), b)