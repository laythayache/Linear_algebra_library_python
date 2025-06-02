# linear_algebra/decompositions.py
from linear_algebra.matrix import Matrix


def covariance_matrix(X: Matrix) -> Matrix:
    rows, cols = X.shape()
    means = [sum(row[i] for row in X.rows) / rows for i in range(cols)]
    centered = Matrix(
        [[x - means[j] for j, x in enumerate(row)] for row in X.rows]
    )
    return centered.transpose().multiply(centered).divide(rows - 1)


def pca_projection(X: Matrix, num_components: int = 2) -> Matrix:
    cov = covariance_matrix(X)
    eigvecs = Matrix.identity(cov.num_rows)  # placeholder eigenvectors
    basis = Matrix([row[:num_components] for row in eigvecs.rows])
    return X.multiply(basis)
