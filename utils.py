from dataclasses import dataclass
import numpy as np

valid_navigation_strategies = [
    'stay_put',
    'thrust_towards_target',
    'line_follow',
    'potential_field',
    'lyapunov_pd',
    'lyapunov_nonlinear',
    'nav_function',
    'chase',
    '_'
]

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

def segment_circle_intersect(A, B, C, r):
    """
    Check if line segment AB intersects a circle with center C and radius r.
    Handles the case where A == B (zero-length segment) safely.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    # Segment vector
    d = B - A
    # Vector from circle center to segment start
    f = A - C

    a = np.dot(d, d)

    # ---- NEW: Handle zero-length segment (A == B) ----
    if a == 0:
        # A and B are the same point.
        # Just check if that point is within the circle.
        return np.linalg.norm(A - C) <= r

    # ---- Normal case: solve quadratic ----
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b*b - 4*a*c

    if discriminant < 0:
        return False  # no intersection

    discriminant_sqrt = np.sqrt(discriminant)
    t1 = (-b - discriminant_sqrt) / (2*a)
    t2 = (-b + discriminant_sqrt) / (2*a)

    # Check if intersection points lie on the segment
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)