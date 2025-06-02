# examples/regression.py
"""
Demonstration: Ordinary-least-squares linear regression.

Fit y = w0 + w1·x on a tiny synthetic dataset.
"""

from linear_algebra.matrix import Matrix
from linear_algebra.solvers import least_squares

# X matrix includes a bias column of 1s
X = Matrix(
    [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
    ]
)
y = Matrix([[1], [1.9], [3.1], [3.9]])

w = least_squares(X, y)  # column vector [[w0], [w1]]

print("Fitted parameters (w0, w1):", [round(val[0], 4) for val in w.rows])

# Predict on x = 5
x5 = Matrix([[1, 5]])
prediction = x5.multiply(w)
print("Prediction for x = 5:", round(prediction.rows[0][0], 4))
