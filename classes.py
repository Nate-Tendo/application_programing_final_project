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
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    d = B - A
    f = A - C

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return False

    discriminant_sqrt = np.sqrt(discriminant)
    t1 = (-b - discriminant_sqrt) / (2*a)
    t2 = (-b + discriminant_sqrt) / (2*a)

    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True

    return False

class Body:
    """Base class for planets, moons, stars, spacecraft."""
    _instances = []
    _index_counter = -999
    
    def __init__(self, name: str, mass: float, position: tuple, velocity = (0,0), color = 'blue', radius = 10, is_dynamically_updated = True):
        self.name = name
        self.mass = mass
        self.position = np.array(position,dtype=float)
        self.velocity = np.array(velocity,dtype=float)
        self.color = color
        self.radius = radius
        self.is_crashed = False
        self.is_dynamically_updated = is_dynamically_updated
        
        # Ensure no overlapping bodies
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
        distance_vector = self.get_relative_position(other)
        distance_magnitude = np.linalg.norm(distance_vector)
        if distance_magnitude < EPSILON_GRAVITY:
            return np.array([0.0, 0.0])
        acceleration_vector = GRAVITY_CONSTANT * other.mass * distance_vector / (distance_magnitude**3)
        return acceleration_vector    
    
    def compute_total_current_acceleration_from_bodies(self):
        accel_vec = sum((self.gravitational_acceleration_from(body)
                    for body in Body._instances if (body is not self) 
                    and (not isinstance(body, Spacecraft))), np.array([0,0]))
        # Note: we intentionally exclude Spacecraft to avoid mutual gravity among ships for now
        return accel_vec
    
    #  TODO: Find a way to simultaneously step bodies forward.
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
        
        # Update orientation toward target for convenience (used by some strategies)
        for ship in Spacecraft._instances:
            if ship.is_target and ship is not self:
                self.orientation = np.arctan2(
                    ship.position[1] - self.position[1],
                    ship.position[0] - self.position[0]
                )

        # gravitational acceleration from bodies (non-spacecraft)
        accel_from_planets = self.compute_total_current_acceleration_from_bodies()

        # ----- Potential Field Navigation Implementation -----
        # This computes a guidance acceleration (desired_accel) using an attractive
        # component toward the target and repulsive components away from bodies.
        ship_thrust_accel = np.array([0.0,0.0])  # default (acceleration due to thrust)
        if self.navigation_strategy == 'potential_field':
            # parameters (tune these)
            k_att = 1.0      # attractive gain (accel per unit distance)
            k_rep = 5000.0   # repulsive gain (scale of repulsion)
            d0 = 80.0        # influence radius of obstacles (any body closer than d0 repels)
            soft = 1e-6

            # find target body (unique)
            target = None
            for ship in Spacecraft._instances:
                if ship.is_target and ship is not self:
                    target = ship
                    break

            if target is None:
                desired_accel = np.array([0.0, 0.0])
            else:
                # Attractive acceleration (pull towards target)
                pos = self.position
                dir_to_goal = target.position - pos
                dist_to_goal = np.linalg.norm(dir_to_goal) + soft
                att_accel = k_att * dir_to_goal  # direction * distance * k_att

                # Repulsive acceleration (push away from each body inside d0)
                rep_accel = np.array([0.0, 0.0])
                for body in Body._instances:
                    if body is self:
                        continue
                    # treat target as not an obstacle for repulsion (optional)
                    if isinstance(body, Spacecraft) and body.is_target:
                        continue
                    vec = pos - body.position
                    dist = np.linalg.norm(vec)
                    if dist < EPSILON_GRAVITY:
                        # If overlapping, create a large immediate repulsion
                        repulse = (vec + 1e-3) * k_rep * 10.0
                        rep_accel += repulse
                        continue
                    if dist <= d0:
                        # unit vector away from obstacle
                        u = vec / dist
                        # magnitude: k_rep * (1/d - 1/d0)
                        mag = k_rep * (1.0/dist - 1.0/d0)
                        rep_accel += u * mag

                # Desired guidance acceleration (combine)
                # Note: att_accel is proportional to distance; rep_accel pushes away strongly near obstacles.
                desired_accel = att_accel + rep_accel

            # Convert desired_accel (m/s^2) to a thrust vector (force): F = m * a
            thrust_needed_vec = desired_accel * self.mass  # force vector
            thrust_needed_mag = np.linalg.norm(thrust_needed_vec)
            if thrust_needed_mag > self.max_thrust and self.max_thrust > 0:
                thrust_vec = thrust_needed_vec / thrust_needed_mag * self.max_thrust
            else:
                thrust_vec = thrust_needed_vec

            # Finally convert thrust vector to acceleration from propulsion (same as thrust_vec / mass)
            ship_thrust_accel = thrust_vec / (self.mass if self.mass != 0 else 1.0)
            # we keep track of fuel_spent by thrust magnitude later as before
            ship_thrust = thrust_vec  # keep the force vector for fuel accounting

        # ----- Existing navigation modes (left unchanged) -----
        elif self.navigation_strategy == 'thrust_towards_target':
            ship_thrust = self.propulsion_acceleration(self.max_thrust, self.orientation) * self.mass  # convert accel->force

            # Here propulsion_acceleration returns acceleration per 1 unit thrust magnitude.
            # We want to use the full max_thrust in direction self.orientation, so convert to force:
            # propulsion_acceleration returns ax = thrust_mag * cos(...) / mass, so multiply by mass to get force vector
            # but simpler approach used above.

            # convert to acceleration
            ship_thrust_accel = ship_thrust / self.mass if self.mass != 0 else np.array([0.0,0.0])

        elif self.navigation_strategy == 'counteract_gravity':
            thrust_towards_target = self.propulsion_acceleration(self.max_thrust, self.orientation) * self.mass
            # Find direction toward target
            if np.linalg.norm(thrust_towards_target) > 0:
                target_unit_vector = thrust_towards_target / np.linalg.norm(thrust_towards_target)
            else:
                target_unit_vector = np.array([0,0])

            gravity_aligned_with_thrust = np.dot(accel_from_planets, target_unit_vector) * target_unit_vector
            ship_thrust_vec = thrust_towards_target - (accel_from_planets * self.mass)
            thrust_mag = np.linalg.norm(ship_thrust_vec)
            if thrust_mag > self.max_thrust:
                ship_thrust_vec = ship_thrust_vec / thrust_mag * self.max_thrust
            ship_thrust_accel = ship_thrust_vec / self.mass if self.mass != 0 else np.array([0.0,0.0])

        else:
            # default: no thrust
            ship_thrust = np.array([0.0,0.0])
            ship_thrust_accel = np.array([0.0,0.0])

        # Track fuel spent:
        # If we have a thrust vector in force-units (N) we approximate fuel use = |thrust| * time
        # Earlier you tracked fuel as (norm(ship_thrust) * self.mass) * time_step; keep similar accounting.
        if 'ship_thrust' in locals():
            fuel_term = np.linalg.norm(ship_thrust) * time_step
        else:
            # ship_thrust might not have been set in potential_field branch; compute from accel
            fuel_term = np.linalg.norm(ship_thrust_accel) * self.mass * time_step

        self.fuel_spent += fuel_term

        # Apply total acceleration (gravity + thrust-accel)
        total_accel = accel_from_planets + ship_thrust_accel
        new_position = self.position + self.velocity*time_step + 0.5*total_accel*(time_step**2)
        
        # Collision detection along the straight-line motion
        for body in Body._instances:
            if body is self:
                continue
            if segment_circle_intersect(self.position, new_position, body.position, body.radius):
                self.is_crashed = True
                print(f"Crash between {self.name} and {body.name}")
        
        self.position = new_position
        new_total_accel_n_plus_1 = self.compute_total_current_acceleration_from_bodies()
        new_velocity = self.velocity + (1/2)*(total_accel + new_total_accel_n_plus_1)*time_step
        self.velocity = new_velocity
        self.path = np.append(self.path, [self.position.copy()], axis=0)

        return None
