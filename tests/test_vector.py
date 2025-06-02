# tests/test_vector.py
from linear_algebra.vector import Vector
from linear_algebra.utils import sqrt


def test_len():
    v = Vector([1, 2, 3])
    assert len(v) == 3


def test_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert v1.dot(v2) == 32


def test_norm():
    v = Vector([3, 4])
    # using custom sqrt to stay consistent with library
    assert abs(v.norm() - 5.0) < 1e-8


def test_add_subtract():
    v1 = Vector([1, 1, 1])
    v2 = Vector([2, 2, 2])
    assert v1.add(v2).values == [3, 3, 3]
    assert v2.subtract(v1).values == [1, 1, 1]


def test_scale_divide():
    v = Vector([2, 4, 6])
    assert v.scale(0.5).values == [1, 2, 3]
    assert v.divide(2).values == [1, 2, 3]


def test_normalize():
    v = Vector([0, 0, 5])
    nv = v.normalize()
    assert abs(nv.norm() - 1.0) < 1e-8
