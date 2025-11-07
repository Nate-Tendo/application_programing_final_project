import numpy as np
from typing import List

G = 6.67430e-11  # gravitational constant

class Vector2:
    """2D vector class backed by numpy for fast vector math."""
    def __init__(self, x: float, y: float):
        self.v = np.array([x, y], dtype=float)

    @property
    def x(self):
        return self.v[0]

    @property
    def y(self):
        return self.v[1]

    def __add__(self, other):
        return Vector2(*(self.v + other.v))

    def __sub__(self, other):
        return Vector2(*(self.v - other.v))

    def __mul__(self, scalar: float):
        return Vector2(*(self.v * scalar))

    def magnitude(self):
        return np.linalg.norm(self.v)

    def normalized(self):
        mag = self.magnitude()
        return Vector2(*(self.v / mag)) if mag > 0 else Vector2(0, 0)

    def copy(self):
        return Vector2(self.x, self.y)

    def __repr__(self):
        return f"Vector2({self.x:.3f}, {self.y:.3f})"

class Body:
    """Base class for planets, moons, stars, spacecraft."""
    _instances = []
    def __init__(self, name: str, mass: float, position: Vector2, velocity: Vector2):
        self.name = name
        self.mass = mass # could add a density and size alternative instead of just mass
        self.position = position
        self.velocity = velocity
        Body._instances.append(self)

    def gravitational_acceleration_from(self, other: "Body") -> Vector2:
        # Newton’s law of gravitation
        direction = other.position - self.position
        distance = direction.magnitude()
        force_mag = G * other.mass / (distance**2)
        return direction.normalized() * force_mag

class SolarSystem:
    """A simple solar system simulator."""
    def __init__(self, name: str, star: Body):
        self.name = name
        self.star = star
        self.bodies: List[Body] = [star]  # 1 star always included

    def add_body(self, body: Body):
        self.bodies.append(body)

    def all_objects(self):
        return self.bodies + self.spacecraft

    def update(self, dt: float):
        objects = self.all_objects()
        accelerations = {obj: Vector2(0, 0) for obj in objects}

        # Gravity between all massive bodies (again, unknown functinality)
        for a in objects:
            for b in objects:
                if a is not b:
                    accelerations[a] = accelerations[a] + a.gravitational_acceleration_from(b)

        # Update positions and velocities (i think... never tested)
        for obj in objects:
            obj.velocity = obj.velocity + accelerations[obj] * dt
            obj.position = obj.position + obj.velocity * dt
