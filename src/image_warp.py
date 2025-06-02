# examples/image_warp.py
"""
Demonstration: 2-D point warp (rotation + translation).

We treat the four corners of a unit square as “image” points,
convert them to homogeneous coordinates, apply a transform,
and print the new positions.
"""

from linear_algebra.transform import rotation_2d, translation_2d, apply_transform_2d
from linear_algebra.matrix import Matrix

# ------------------------------------------------------------
# 1. original corner points in homogeneous coords [x, y, 1]
square_pts = [
    [0, 0, 1],  # top-left
    [1, 0, 1],  # top-right
    [1, 1, 1],  # bottom-right
    [0, 1, 1],  # bottom-left
]

# 2. build a composite transform: rotate 45° (≈ 0.7854 rad) then translate (2, 1)
rot = rotation_2d(0.785398)           # 45 °
trans = translation_2d(2, 1)
composite = trans.multiply(rot)       # first rotate, then translate

# 3. apply to every point
warped = [apply_transform_2d(composite, p) for p in square_pts]

print("Original points:")
for p in square_pts:
    print(p[:2])

print("\nWarped points:")
for p in warped:
    print([round(c, 3) for c in p[:2]])
