# examples/pca_demo.py
"""
Demonstration: Very small PCA.

Uses identity eigenvectors (placeholder) just to
show centring and projection workflow.
"""

from linear_algebra.matrix import Matrix
from linear_algebra.decompositions import covariance_matrix, pca_projection

# 4 samples, 3 features
X = Matrix(
    [
        [2.5, 2.4, 0.5],
        [0.5, 0.7, 1.0],
        [2.2, 2.9, 0.9],
        [1.9, 2.2, 1.3],
    ]
)

print("Original data:")
for row in X.rows:
    print(row)

cov = covariance_matrix(X)
print("\nCovariance matrix:")
for row in cov.rows:
    print([round(v, 4) for v in row])

proj = pca_projection(X, num_components=2)  # projects to first 2 “components”
print("\nProjected data (2-D):")
for row in proj.rows:
    print([round(v, 4) for v in row])
