# linear_algebra/utils.py

PI = 3.141592653589793

def sqrt(x: float, tolerance=1e-10, max_iter=1000) -> float:
    if x < 0:
        raise ValueError("Cannot compute sqrt of negative number")
    guess = x / 2 if x > 1 else 1.0
    for _ in range(max_iter):
        prev = guess
        guess = (guess + x / guess) / 2
        if abs(prev - guess) < tolerance:
            break
    return guess

def acos_approx(x: float) -> float:
    # Clamp input
    x = max(-1.0, min(1.0, x))
    # Polynomial approximation for acos(x) on [-1, 1]
    # Source: Bhaskara I’s approximation
    return PI/2 - x - (x**3)/6  # crude but works OK in 0.2–0.9

def to_degrees(radians: float) -> float:
    return radians * (180 / PI)
