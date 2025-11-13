import numpy as np
from typing import List
from dataclasses import dataclass


GRAVITY_CONSTANT = 1
EPSILON_GRAVITY = 1e-8

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

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

class Body:
    """Base class for planets, moons, stars, spacecraft."""
    _instances = []
    _index_counter = -999
    # TODO: Make it easier to initialize velocity at 0 (automatically convert 0 -> Vector2(0,0))
    
    def __init__(self, name: str, mass: float, position: list, velocity = (0,0), color = 'blue', radius = 10, is_dynamically_updated = True):
        self.name = name
        self.mass = mass # could add a density and size alternative instead of just mass
        self.position = np.array(position,dtype=float)
        self.velocity = np.array(velocity,dtype=float)
        self.color = color
        self.radius = radius
        self.is_crashed = False
        self.is_dynamically_updated = is_dynamically_updated
        
        # Check to ensure new body does not creat inteference!
        for element in Body._instances:
            distance_separation = np.linalg.norm((element.position - self.position))
            if distance_separation < (self.radius +  element.radius):
                raise ValueError(f"New Body ({self.name}) Interferes with {element.name}, Cannot Proceed")
        
        Body._instances.append(self)
        if len(Body._instances) == 0:
            Body._index_counter = 0
        else:
            Body._index_counter += 1

    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]
    
    @property
    def vx(self):
        return self.velocity[0]
    
    @property
    def vy(self):
        return self.velocity[1]
         
    def get_relative_position(self, other: 'Body') -> np.array:
        return other.position - self.position
    
    def get_relative_speed(self, other: 'Body') -> np.array:
        return other.velocity - self.velocity

    def gravitational_acceleration_from(self, other: 'Body') -> np.array:
        # Newton’s law of gravitation
        distance_vector = other.position - self.position                                                 # Position of body from the perspective of self, these are numpy arrays
        distance_magnitude = np.linalg.norm(distance_vector)                                             # This will be a scalar
        acceleration_vector = GRAVITY_CONSTANT * other.mass * distance_vector / (distance_magnitude**3)  # Derived from m1a = Gm1m2/d^2, note how m1 cancels.
        return acceleration_vector    
    
    def compute_total_current_acceleration_from_bodies(self):
        accel_vec = sum((self.gravitational_acceleration_from(body)
                    for body in Body._instances if (body is not self) 
                    and (not isinstance(body, Spacecraft))), np.array([0,0]))
        return accel_vec
    
    def step_forward_dt(self, time_step = 0.1):
        # Use Velocity Verlet Numerical Integration
        total_accel = self.compute_total_current_acceleration_from_bodies()
        new_position = self.position + self.velocity*time_step + (1/2)*total_accel*(time_step**2)
        self.position = new_position
        new_total_accel_n_plus_1 = self.compute_total_current_acceleration_from_bodies()
        new_velocity = self.velocity + (1/2)*(total_accel + new_total_accel_n_plus_1)*time_step
        self.velocity = new_velocity
        return

class Spacecraft(Body):

    _instances = []
    _index_counter = -999

    def __init__(self, name, mass, position, velocity = (0,0), color = 'white', thrust=0.0, orientation=0.0, radius = 1, is_dynamically_updated = True, is_target = False):
        super().__init__(name, mass, position, velocity, color, radius, is_dynamically_updated)
        self.max_thrust = thrust
        self.orientation = orientation  # radians
        self.path = np.array([self.position.copy()])  # Store the path as an array of positions
        self.list_boosters_on = {'up': 0,'down':0, 'left': 0, 'right': 0}
        self.fuel_spent = 0.0
        self.is_target = is_target
        self.navigation_strategy = 'none'  

        if self.name == 'target' and not self.is_target:
            raise ValueError("Spacecraft named 'target' must have is_target=True")
        
        if self.is_target and any(ship.name == 'target' and ship is not self for ship in Spacecraft._instances):
            raise ValueError("There can only be one target spacecraft")

        Spacecraft._instances.append(self)
        if len(Spacecraft._instances) == 0:
            Spacecraft._index_counter = 0
        else:
            Spacecraft._index_counter += 1

    # TDOO: Update propulsion to take in orietnation relative to spacecraft and translate to the environment
    def propulsion_acceleration(self,thrust_magnitude,thrust_direction):
        ax = thrust_magnitude * np.cos(thrust_direction) / self.mass
        ay = thrust_magnitude * np.sin(thrust_direction) / self.mass
        return np.array([ax,ay])
    
    def step_forward_dt(self, time_step = 0.1):
        
        # Use Velocity Verlet Numerical Integration
        for ship in Spacecraft._instances:
            if ship.is_target and ship is not self:
                self.orientation = np.arctan2(
                    ship.position[1] - self.position[1],
                    ship.position[0] - self.position[0]
                )

        accel_from_planets = self.compute_total_current_acceleration_from_bodies()

        if self.navigation_strategy == 'thrust_towards_target':
            ship_thrust = self.propulsion_acceleration(self.max_thrust, self.orientation)

        elif self.navigation_strategy == 'counteract_gravity':
            thrust_towards_target = self.propulsion_acceleration(self.max_thrust * .2, self.orientation)

            # Find direction toward target
            if np.linalg.norm(thrust_towards_target) > 0:
                target_unit_vector = thrust_towards_target / np.linalg.norm(thrust_towards_target)
            else:
                target_unit_vector = np.array([0,0])

            # Compute gravity aligned with thrust
            gravity_aligned_with_thrust = np.dot(accel_from_planets, target_unit_vector) * target_unit_vector

            ship_thrust = thrust_towards_target - (accel_from_planets - gravity_aligned_with_thrust)

            thrust_mag = np.linalg.norm(ship_thrust)
            if thrust_mag > self.max_thrust:
                ship_thrust = ship_thrust / thrust_mag * self.max_thrust  # rescale
        else:
            ship_thrust=np.array((0,0))

        # Track fuel spent
        self.fuel_spent += (np.linalg.norm(ship_thrust) * self.mass) * time_step

        # Apply total acceleration (gravity + thrust)
        total_accel = accel_from_planets + ship_thrust
        new_position = self.position + self.velocity*time_step + 0.5*total_accel*(time_step**2)
        
        # BEFORE WE SET THSI NEW POSITION, WE WANT TO CHECK IF WE'VE HIT ANYTHING..
        # Taking some inspiration from Chat-GPT, but this is going to be quite computationally expensive...
        for body in Body._instances:
            if body is self:
                continue
            if segment_circle_intersect(self.position, new_position, body.position, body.radius): #Remember that position.v is the position vector!
                self.is_crashed = True
                print(f"Crash between {self.name} and {body.name}")
        
        self.position = new_position
        new_total_accel_n_plus_1 = self.compute_total_current_acceleration_from_bodies()
        new_velocity = self.velocity + (1/2)*(total_accel + new_total_accel_n_plus_1)*time_step
        self.velocity = new_velocity
        self.path = np.append(self.path, [self.position.copy()], axis=0)

        return None