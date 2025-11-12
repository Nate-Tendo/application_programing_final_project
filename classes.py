import numpy as np
from typing import List

# GRAVITY_CONSTANT = 6.67430e-11  # gravitational constant --> Although this is the true G, we need to scale it for realism
GRAVITY_CONSTANT = 1
EPSILON_GRAVITY = 1e-8

def segment_circle_intersect(A, B, C, r): # This function definition written with ChatGPT's help as the math for line segment into a circle gets messy
    """
    Check if line segment AB intersects a circle with center C and radius r.

    Parameters
    ----------
    A, B, C : array-like or numpy arrays of shape (2,)
        Coordinates of the points.
    r : float
        Radius of the circle.

    Returns
    -------
    bool
        True if the segment intersects the circle, False otherwise.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    # Segment vector
    d = B - A
    # Vector from circle center to segment start
    f = A - C

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        # No intersection
        return False

    # Quadratic formula
    discriminant_sqrt = np.sqrt(discriminant)
    t1 = (-b - discriminant_sqrt) / (2*a)
    t2 = (-b + discriminant_sqrt) / (2*a)

    # Check if either t is within the segment [0,1]
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True

    return False

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
        if other == 0:
            return self
        return Vector2(*(self.v + other.v))
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return Vector2(*(self.v - other.v))
    
    # TODO
    # Def __rsub__

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
    
    def __init__(self, name: str, mass: float, position: Vector2, velocity: Vector2, color = 'blue', radius = 10):
        self.name = name
        self.mass = mass # could add a density and size alternative instead of just mass
        self.position = position
        self.velocity = velocity
        self.color = color
        self.radius = radius
        
        # Check to ensure new body does not creat inteference!
        for element in Body._instances:
            distance_separation = (element.position - self.position).magnitude()
            if distance_separation < (self.radius +  element.radius):
                raise ValueError(f"New Body ({self.name}) Interferes with {element.name}, Cannot Proceed")
        
        Body._instances.append(self)
         
    def get_relative_position(self, other: 'Body') -> Vector2:
        return other.position - self.position
    
    def get_relative_speed(self, other: 'Body') -> Vector2:
        return other.velocity - self.velocity

    def gravitational_acceleration_from(self, other: 'Body') -> Vector2:
        # Newton’s law of gravitation
        direction = self.get_relative_position(other)
        distance = direction.magnitude()
        force_mag = GRAVITY_CONSTANT * other.mass / (distance**2 + EPSILON_GRAVITY**2)
        return direction.normalized() * force_mag

class Spacecraft(Body):

    _instances = []

    def __init__(self, name, mass, position, velocity, color, thrust=0.0, orientation=0.0, radius = 1):
        super().__init__(name, mass, position, velocity, color, radius)
        self.thrust = thrust
        self.orientation = orientation  # radians
        self.path = [self.position.copy()]
        self.list_boosters_on = {'up': 0,'down':0, 'left': 0, 'right': 0}

        Spacecraft._instances.append(self)

    # TDOO: Update propulsion to take in orietnation relative to spacecraft and translate to the environment
    def propulsion_acceleration(self,thrust_direction):
        ax = self.thrust * np.cos(thrust_direction) / self.mass
        ay = self.thrust * np.sin(thrust_direction) / self.mass
        return Vector2(ax, ay)
    
    def compute_total_current_force(self):
        
        force_bodies = sum([self.gravitational_acceleration_from(body) for body in Body._instances if not isinstance(body,Spacecraft)])
        
        
        ## VERY SIMPLE FORCE-BOOSTING SCHEME. Will definitely need to update and probably translate into an appropriate frame
        force_boosters = 0
        
        if self.list_boosters_on['up'] == 1:
            print('up')
            force_boosters += self.propulsion_acceleration(np.deg2rad(90))
            self.list_boosters_on['up'] = 0
        if self.list_boosters_on['down'] == 1:
            print('down')
            force_boosters += self.propulsion_acceleration(np.deg2rad(-90))
            self.list_boosters_on['down'] = 0
            
        if self.list_boosters_on['left'] == 1:
            print('left')
            force_boosters += self.propulsion_acceleration(np.deg2rad(180))
            self.list_boosters_on['left'] = 0
            
        if self.list_boosters_on['right'] == 1:
            print('right')
            force_boosters += self.propulsion_acceleration(np.deg2rad(0))
            self.list_boosters_on['right'] = 0
        
        return force_bodies + force_boosters
    
    def step_forward_dt(self, time_step = 0.1):
        body_crash = False
        
        # Use Velocity Verlet Numerical Integration
        total_force = self.compute_total_current_force()
        new_position = self.position + self.velocity*time_step + (1/2)*total_force*(time_step**2)
        
        # BEFORE WE SET THSI NEW POSITION, WE WANT TO CHECK IF WE'VE HIT ANYTHING..
        # Taking some inspiration from Chat-GPT, but this is going to be quite computationally expensive...
        for body in Body._instances:
            if body is self:
                continue
            if segment_circle_intersect(self.position.v, new_position.v, body.position.v, body.radius): #Remember that position.v is the position vector!
                body_crash = True
                print(f"Crash between {self.name} and {body.name}")
        
        self.position = new_position
        new_total_force_n_plus_1 = self.compute_total_current_force()
        new_velocity = self.velocity + (1/2)*(total_force + new_total_force_n_plus_1)*time_step
        self.velocity = new_velocity
        self.path.append(self.position.copy())
        return body_crash