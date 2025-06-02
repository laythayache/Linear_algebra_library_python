from linear_algebra.vector import Vector


class Matrix:
    def __init__(self, rows: list[list[float]]):
        if not all(isinstance(row, list) for row in rows):
            raise TypeError("Matrix rows must be lists.")
        if len(rows) == 0 or len(rows[0]) == 0:
            raise ValueError("Matrix must have non-zero dimensions.")
        if not all(len(row) == len(rows[0]) for row in rows):
            raise ValueError("All rows must have the same number of columns.")
        self.rows = rows
        self.num_rows = len(rows)
        self.num_cols = len(rows[0])

    def __repr__(self):
        return f"Matrix({self.rows})"

    def __getitem__(self, index: int) -> list[float]:
        return self.rows[index]

    def __len__(self):
        return self.num_rows

    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    def add(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([
            [a + b for a, b in zip(r1, r2)]
            for r1, r2 in zip(self.rows, other.rows)
        ])

    def subtract(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([
            [a - b for a, b in zip(r1, r2)]
            for r1, r2 in zip(self.rows, other.rows)
        ])

    def scale(self, scalar: float) -> "Matrix":
        return Matrix([[x * scalar for x in row] for row in self.rows])

    def divide(self, scalar: float) -> "Matrix":
        if scalar == 0:
            raise ZeroDivisionError("Division by zero.")
        return Matrix([[x / scalar for x in row] for row in self.rows])

    def transpose(self) -> "Matrix":
        transposed = [[self.rows[r][c] for r in range(self.num_rows)] for c in range(self.num_cols)]
        return Matrix(transposed)

    def multiply(self, other: "Matrix") -> "Matrix":
        if self.num_cols != other.num_rows:
            raise ValueError("Matrix dimensions do not match for multiplication.")
        result = []
        for i in range(self.num_rows):
            row = []
            for j in range(other.num_cols):
                col = [other.rows[k][j] for k in range(other.num_rows)]
                val = sum(self.rows[i][k] * col[k] for k in range(self.num_cols))
                row.append(val)
            result.append(row)
        return Matrix(result)

    @staticmethod
    def identity(size: int) -> "Matrix":
        return Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    @staticmethod
    def from_vector(vector: Vector, as_row: bool = True) -> "Matrix":
        if as_row:
            return Matrix([vector.values])
        else:
            return Matrix([[v] for v in vector.values])

    def _check_same_shape(self, other: "Matrix"):
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape.")
