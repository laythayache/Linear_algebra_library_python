# linear_algebra/transform.py
from linear_algebra.matrix import Matrix
from linear_algebra.utils import sin_approx, cos_approx


def rotation_2d(theta_radians: float) -> Matrix:
    cos = cos_approx(theta_radians)
    sin = sin_approx(theta_radians)
    return Matrix(
        [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ]
    )


def scaling_2d(sx: float, sy: float) -> Matrix:
    return Matrix(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ]
    )


def translation_2d(tx: float, ty: float) -> Matrix:
    return Matrix(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )


def apply_transform_2d(transform: Matrix, point: list[float]) -> list[float]:
    if len(point) != 3:
        raise ValueError("Point must be [x, y, 1] in homogeneous coords.")
    result = transform.multiply(Matrix([point]).transpose())
    return [row[0] for row in result.transpose().rows]
