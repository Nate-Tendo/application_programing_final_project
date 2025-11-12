import numpy as np
import random
from dataclasses import dataclass
import json

GRAVITY_CONSTANT = 1

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

def load_universe_from_config(config_path: str):
    """Load Universe Details from config file"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load bounds
    b = config.get("bounds", {})
    bounds = Bounds(
        b.get("x_min", -1e9),
        b.get("x_max", 1e9),
        b.get("y_min", -1e9),
        b.get("y_max", 1e9),
    )

    # Create Universe
    universe = Universe(bounds=bounds)

    # Load celestial bodies
    for body_cfg in config.get("bodies", []):
        body = CelestialBody(
            name=body_cfg["name"],
            mass=body_cfg["mass"],
            position=tuple(body_cfg["position"]),
            velocity=tuple(body_cfg.get("velocity", [0, 0])),
            radius=body_cfg["radius"],
            color=body_cfg.get("color", "white"),
        )
        universe.add_body(body)

    # Load dynamic ships
    for ship_cfg in config.get("ships", []):
        ship = VehicleBody(
            name=ship_cfg["name"],
            mass=ship_cfg["mass"],
            position=tuple(ship_cfg["position"]),
            velocity=tuple(ship_cfg.get("velocity", [0, 0])),
            radius=ship_cfg["radius"],
            color=ship_cfg.get("color", "white"),
            max_thrust=ship_cfg.get("max_thrust", np.inf),
        )
        universe.add_vehicle(ship)

    return universe

class CelestialBody:
    """Base class for planets, moons, stars, spacecraft."""
    
    def __init__(self, name, mass, position, velocity, radius, color = 'white'):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.radius = radius
        self.is_dynamic = False # TODO: How do we want to represent this?
        
    def gravitational_acceleration_from_body(self, body):
        # Newton’s law of gravitation
        distance_vector = body.position - self.position                                                 # Position of body from the perspective of self, these are numpy arrays
        distance_magnitude = np.linalg.norm(distance_vector)                                            # This will be a scalar
        acceleration_vector = GRAVITY_CONSTANT * body.mass * distance_vector / (distance_magnitude**3)  # Derived from m1a = Gm1m2/d^2, note how m1 cancels.
        return acceleration_vector                                          

class VehicleBody(CelestialBody):
    def __init__(self, name, mass, position, velocity, radius, color = 'red', orientation=0.0, max_thrust = np.inf):
        super().__init__(name, mass, position, velocity, radius, color)
        self.orientation = orientation                      # radians
        self.trail = [self.position.copy()]
        self.max_thrust = max_thrust
        self.thrust_on = False
        self.list_boosters_on = {'up': 0,'down':0, 'left': 0, 'right': 0}

    def update_trail(self):
        if len(self.trail) > 2000:
            self.trail.pop(0)
        self.trail.append(self.position.copy())
    
    def propulsion_acceleration(self,thrust_magnitude,thrust_direction):
        ax = thrust_magnitude * np.cos(thrust_direction) / self.mass
        ay = thrust_magnitude * np.sin(thrust_direction) / self.mass
        return np.array([ax,ay])
    
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

class PhysicsEngine:
    

    @staticmethod
    def gravitational_acceleration(pos, bodies):
        total = np.zeros(2)
        for b in bodies:
            r_vec = b.position - pos
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue
            total += PhysicsEngine.G * b.mass * r_vec / (r ** 3)
        return total

    @staticmethod
    def integrate_rk4(ship, bodies, dt, thrust=np.zeros(2)):
        def accel(pos, vel):
            return PhysicsEngine.gravitational_acceleration(pos, bodies) + thrust / ship.mass

        x0, v0 = ship.position.copy(), ship.velocity.copy()
        a1 = accel(x0, v0)
        k1v = a1 * dt
        k1x = v0 * dt

        a2 = accel(x0 + 0.5 * k1x, v0 + 0.5 * k1v)
        k2v = a2 * dt
        k2x = (v0 + 0.5 * k1v) * dt

        a3 = accel(x0 + 0.5 * k2x, v0 + 0.5 * k2v)
        k3v = a3 * dt
        k3x = (v0 + 0.5 * k2v) * dt

        a4 = accel(x0 + k3x, v0 + k3v)
        k4v = a4 * dt
        k4x = (v0 + k3v) * dt

        ship.velocity = v0 + (k1v + 2*k2v + 2*k3v + k4v) / 6
        ship.position = x0 + (k1x + 2*k2x + 2*k3x + k4x) / 6


class Universe:
    def __init__(self, num_bodies = 0, bounds= Bounds(-1e9, 1e9, -1e9, 1e9)):
        self.bounds = bounds
        self.all_bodies = []
        self.ships = []
        if num_bodies > 0:
            self.generate_bodies_randomly(num_bodies)

        

    def generate_bodies_randomly(self, n):
        xmin, xmax, ymin, ymax = self.bounds
        for i in range(n):
            pos = [random.uniform(xmin, xmax), random.uniform(ymin, ymax)]
            mass = random.uniform(1e23, 1e26)
            radius = random.uniform(5e6, 2e7)
            color = random.choice(['yellow', 'orange', 'blue', 'red'])
            self.bodies.append(CelestialBody(f"Body{i}", mass, pos, radius, color))

    def add_body(self,new_body):
        for element in self.all_bodies:
            print(new_body.position)
            print(element.position)
            distance_separation = abs(np.linalg.norm(new_body.position - element.position))
            if distance_separation < (new_body.radius +  element.radius):
                raise ValueError(f"New Body ({new_body.name}) Interferes with {element.name}, Cannot Proceed")
        
        self.all_bodies.append(new_body)

    def add_vehicle(self,new_ship):
        for element in self.all_bodies:
            distance_separation = abs(np.linalg.norm(new_ship.position - element.position))
            if distance_separation < (new_ship.radius +  element.radius):
                raise ValueError(f"New Body ({new_ship.name}) Interferes with {element.name}, Cannot Proceed")
        self.ships.append(new_ship)



class TrajectoryPredictor:
    """Predicts trajectory under gravity + thrust guidance toward goal."""
    def __init__(self, universe):
        self.universe = universe

    def predict(self, ship, goal, dt, steps, thrust_mag):
        s = DynamicBody(ship.name + "_pred", ship.position.copy(), ship.velocity.copy(), ship.mass)
        predicted = [s.position.copy()]

        for _ in range(steps):
            to_goal = goal - s.position
            dist = np.linalg.norm(to_goal)
            if dist < 1e7:
                break
            dir_to_goal = to_goal / dist
            thrust = thrust_mag * dir_to_goal
            PhysicsEngine.integrate_rk4(s, self.universe.bodies, dt, thrust)
            predicted.append(s.position.copy())
        return np.array(predicted)
    