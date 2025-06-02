# tests/test_matrix.py
from linear_algebra.matrix import Matrix


def test_shape_and_len():
    A = Matrix([[1, 2], [3, 4], [5, 6]])
    assert len(A) == 3
    assert A.shape() == (3, 2)


def test_add_subtract():
    A = Matrix([[1, 1], [1, 1]])
    B = Matrix([[2, 2], [2, 2]])
    C = A.add(B)
    D = B.subtract(A)
    assert C.rows == [[3, 3], [3, 3]]
    assert D.rows == [[1, 1], [1, 1]]


def test_scale_divide():
    A = Matrix([[2, 4], [6, 8]])
    assert A.scale(0.5).rows == [[1, 2], [3, 4]]
    assert A.divide(2).rows == [[1, 2], [3, 4]]


def test_transpose():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    At = A.transpose()
    assert At.rows == [[1, 4], [2, 5], [3, 6]]


def test_identity_multiply():
    I = Matrix.identity(2)
    A = Matrix([[7, 8], [9, 10]])
    assert I.multiply(A).rows == A.rows
    assert A.multiply(I).rows == A.rows


def test_matrix_multiply():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[7, 8], [9, 10], [11, 12]])
    C = A.multiply(B)
    assert C.rows == [
        [58, 64],
        [139, 154],
    ]
