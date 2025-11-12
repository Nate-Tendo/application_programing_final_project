import numpy as np
from typing import List


# GRAVITY_CONSTANT = 6.67430e-11  # gravitational constant --> Although this is the true G, we need to scale it for realism
GRAVITY_CONSTANT = 1

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
    
    @property
    def angle(self):
        return np.atan2(self.y,self.x)
    
    def __add__(self, other):
        return Vector2(*(self.v + other.v))
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return Vector2(*(self.v - other.v))
    
    def __rsub__(self,other):
        return Vector2(*(other.v-self.v))

    def __mul__(self, scalar: float):
        return Vector2(*(self.v * scalar))
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)  # handles float * Vector2

    def magnitude(self):
        return np.linalg.norm(self.v)

    def normalized(self):  # Could rename to be 'unit_vect?"
        mag = self.magnitude()
        return Vector2(*(self.v / mag)) if mag > 0 else Vector2(0, 0)

    def copy(self):
        return Vector2(self.x, self.y)

    def __repr__(self):
        return f"Vector2({self.x:.6f}, {self.y:.6f})" # Needed more decimals for a smaller time step change

class Body:
    """Base class for planets, moons, stars, spacecraft."""
    _instances = []
    
    # TODO: Make it easier to initialize velocity at 0 (automatically convert 0 -> Vector2(0,0))
    
    def __init__(self, name: str, mass: float, position: Vector2, velocity: Vector2, color = 'blue', state = 'static'):
        self.name = name
        self.mass = mass # could add a density and size alternative instead of just mass
        self.position = position
        self.velocity = velocity
        self.color = color
        self.state = state
        Body._instances.append(self)

    def gravitational_acceleration_from(self, other: "Body") -> Vector2:
        # Newton’s law of gravitation
        direction = other.position - self.position
        distance = direction.magnitude()
        if distance == 0:
            return Vector2(0, 0)
        force_mag = GRAVITY_CONSTANT * other.mass / (distance**2)
        return direction.normalized() * force_mag
    
    def compute_total_current_force(self):
        force_vec = sum((self.gravitational_acceleration_from(body)
                   for body in Body._instances if (body is not self) 
                   and (not isinstance(body, Spacecraft))), Vector2(0, 0))
        return force_vec
    
    def step_forward_dt(self, time_step = 0.1):
        # Use Velocity Verlet Numerical Integration
        total_force = self.compute_total_current_force()
        new_position = self.position + self.velocity*time_step + (1/2)*total_force*(time_step**2)
        self.position = new_position
        new_total_force_n_plus_1 = self.compute_total_current_force()
        new_velocity = self.velocity + (1/2)*(total_force + new_total_force_n_plus_1)*time_step
        self.velocity = new_velocity
        return
    

# for body in Body._instances:
class Spacecraft(Body):
    def __init__(self, name, mass, position, velocity, color, thrust=0.0, orientation=0.0):
        super().__init__(name, mass, position, velocity, color)
        self.thrust = thrust
        self.orientation = orientation  # radians
        self.path = [self.position.copy()]

    def propulsion_acceleration(self):
        ax = self.thrust * np.cos(self.orientation) / self.mass
        ay = self.thrust * np.sin(self.orientation) / self.mass
        return Vector2(ax, ay)
    
    def compute_total_current_force(self):
        return sum([self.gravitational_acceleration_from(body) for body in Body._instances if not isinstance(body,Spacecraft)])
    
    def step_forward_dt(self, time_step = 0.1):
        # Use Velocity Verlet Numerical Integration
        total_force = self.compute_total_current_force()
        new_position = self.position + self.velocity*time_step + (1/2)*total_force*(time_step**2)
        self.position = new_position
        new_total_force_n_plus_1 = self.compute_total_current_force()
        new_velocity = self.velocity + (1/2)*(total_force + new_total_force_n_plus_1)*time_step
        self.velocity = new_velocity
        self.path.append(self.position.copy())
        return
    
    
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