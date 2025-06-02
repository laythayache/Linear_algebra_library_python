# linear_algebra/solvers.py
from linear_algebra.matrix import Matrix


def gaussian_elimination(A: Matrix, b: Matrix) -> Matrix:
    n = len(A)
    Ab = [A[i] + b[i] for i in range(n)]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(Ab[r][i]))
        Ab[i], Ab[pivot] = Ab[pivot], Ab[i]

        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            Ab[j] = [a - factor * b for a, b in zip(Ab[j], Ab[i])]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (
            Ab[i][-1] - sum(Ab[i][j] * x[j] for j in range(i + 1, n))
        ) / Ab[i][i]

    return Matrix([[xi] for xi in x])


def least_squares(X: Matrix, y: Matrix) -> Matrix:
    Xt = X.transpose()
    XtX = Xt.multiply(X)
    Xty = Xt.multiply(y)
    return gaussian_elimination(XtX, Xty)
