from linear_algebra.utils import sqrt, acos_approx, to_degrees


class Vector:
    def __init__(self, values: list[float]):
        if not isinstance(values, list):
            raise TypeError("Values must be a list of floats or ints.")
        if not all(isinstance(x, (int, float)) for x in values):
            raise ValueError("All elements must be int or float.")
        self.values = values
        self.dim = len(values)

    def __repr__(self):
        return f"Vector({self.values})"      ##Returns a string representation of the vector.

    def __len__(self):
        return self.dim

    def dot(self, other: "Vector") -> float:
        self._check_same_dim(other)
        return sum(a * b for a, b in zip(self.values, other.values))

    def norm(self) -> float:
        return sqrt(sum(x**2 for x in self.values))

    def normalize(self) -> "Vector":
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize a zero vector.")
        return Vector([x / n for x in self.values])

    def scale(self, scalar: float) -> "Vector":
        return Vector([x * scalar for x in self.values])

    def divide(self, scalar: float) -> "Vector":
        if scalar == 0:
            raise ZeroDivisionError("Division by zero.")
        return Vector([x / scalar for x in self.values])

    def add(self, other: "Vector") -> "Vector":
        self._check_same_dim(other)
        return Vector([a + b for a, b in zip(self.values, other.values)])

    def subtract(self, other: "Vector") -> "Vector":
        self._check_same_dim(other)
        return Vector([a - b for a, b in zip(self.values, other.values)])

    def angle_with(self, other: "Vector", in_degrees: bool = False) -> float:
        self._check_same_dim(other)
        dot = self.dot(other)
        norms = self.norm() * other.norm()
        if norms == 0:
            raise ValueError("Cannot compute angle with zero vector.")
        cos_theta = max(-1.0, min(1.0, dot / norms))  # Clamp for safety
        angle = acos_approx(cos_theta)
        return to_degrees(angle) if in_degrees else angle

    def project_onto(self, other: "Vector") -> "Vector":
        self._check_same_dim(other)
        other_norm_sq = other.norm() ** 2
        if other_norm_sq == 0:
            raise ValueError("Cannot project onto a zero vector.")
        scalar = self.dot(other) / other_norm_sq
        return other.scale(scalar)

    def _check_same_dim(self, other: "Vector"):
        if not isinstance(other, Vector):
            raise TypeError("Argument must be of type Vector.")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension.")
