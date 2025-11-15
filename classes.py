import numpy as np
from typing import List
from dataclasses import dataclass
import math

GRAVITY_CONSTANT = 1
EPSILON_GRAVITY = 1e-8

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

class Body:
    """Base class for planets, moons, stars, spacecraft."""
    _instances = []
    _index_counter = -999
    # TODO: Make it easier to initialize velocity at 0 (automatically convert 0 -> Vector2(0,0))
    
    def __init__(self, name: str, mass: float, position: tuple, velocity = (0,0), color = 'blue', radius = 10, is_dynamically_updated = True, velocity_vec = False):
        self.name = name
        self.mass = mass # could add a density and size alternative instead of just mass
        self.position = np.array(position,dtype=float)
        self.velocity = np.array(velocity,dtype=float)
        self.color = color
        self.radius = radius
        self.is_crashed = False
        self.is_dynamically_updated = is_dynamically_updated
        
        self.velocity_vec = velocity_vec
        
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
    
    @property
    def vmag(self):
        return np.linalg.norm(self.velocity)
         
    def get_relative_position(self, other: 'Body') -> np.array:
        return other.position - self.position
    
    def get_relative_speed(self, other: 'Body') -> np.array:
        return other.velocity - self.velocity

    # Computes the gravitational acceleration from one body on another
    def gravitational_acceleration_from(self, other: 'Body') -> np.array: 
        # Newton’s law of gravitation
        distance_vector = self.get_relative_position(other)                                                # Position of body from the perspective of self, these are numpy arrays
        distance_magnitude = np.linalg.norm(distance_vector)                                             # This will be a scalar
        acceleration_vector = GRAVITY_CONSTANT * other.mass * distance_vector / (distance_magnitude**3)  # Derived from m1a = Gm1m2/d^2, note how m1 cancels.
        return acceleration_vector    

    # We used ChatGPT for suggestions on this function to reduce function calls and help us figure out the linear algebra.
    @staticmethod
    def compute_gravity_from_sources_to_targets(targets, sources):
        if len(targets) == 0 or len(sources) == 0:
            return np.zeros((len(targets), 2), dtype=float)

        pos_t = np.array([b.position for b in targets])   # (N_t, 2)
        pos_s = np.array([b.position for b in sources])   # (N_s, 2)
        masses_s = np.array([b.mass for b in sources])    # (N_s,)

        disp = pos_s[None, :, :] - pos_t[:, None, :]      # (N_t, N_s, 2)
        dist2 = np.sum(disp**2, axis=2) + EPSILON_GRAVITY
        inv_dist3 = dist2 ** (-1.5)
        accel_contrib = GRAVITY_CONSTANT * disp * masses_s[None, :, None] * inv_dist3[:, :, None]
        accel = np.sum(accel_contrib, axis=1)
        return accel

    # Computes the gravitational acceleration from all other bodies on one body
    def g_from_bodies(self):
        accel_vec = sum((self.gravitational_acceleration_from(body)
                    for body in Body._instances if (body is not self) 
                    and (not isinstance(body, Spacecraft))), np.array([0,0]))
        return accel_vec
    
    # Velocity Verlet Numerical Integration -  computes each step for all bodies to avoid violating conservation of momentum
    def step_forward_dt(self, time_step = 0.1):
        total_accel = self.compute_total_current_acceleration_from_bodies()
        new_position = self.position + self.velocity*time_step + (1/2)*total_accel*(time_step**2)
        self.position = new_position
        new_total_accel_n_plus_1 = self.compute_total_current_acceleration_from_bodies()
        new_velocity = self.velocity + (1/2)*(total_accel + new_total_accel_n_plus_1)*time_step
        self.velocity = new_velocity
        return
    
    @staticmethod
    def timestep(time_step = 1):
        sources = [b for b in Body._instances if not isinstance(b, Spacecraft)]
        
        dynamic_bodies = [body for body in Body._instances if body.is_dynamically_updated]
        gravity_array = Body.compute_gravity_from_sources_to_targets(dynamic_bodies, sources)

        N = len(dynamic_bodies)
        grav_accel = np.zeros((N, 2))     # gravitational acceleration for each body
        total_accel = np.zeros((N, 2))    # total acceleration including thrust
        thrusts = np.zeros((N, 2))        # thrust for each spacecraft


        # compute accelerations at t_n
        for i, b in enumerate(dynamic_bodies):
            grav_accel[i] = gravity_array[i]
            if isinstance(b,Spacecraft):
                thrusts[i] = b.thrust_ctrl_law(grav_accel[i])
                # b.thrust = thrusts[i] <-- Can remove, not used...
                b.thrust_mag = np.linalg.norm(thrusts[i])
                b.fuel_spent += (np.linalg.norm(thrusts[i]) * b.mass) * time_step
                # print(f"{b.name}: {b.fuel_spent}")
                total_accel[i] = grav_accel[i] + thrusts[i]
            else:
                total_accel[i] = grav_accel[i]
        
        # compute all new positions at t_n+1
        for i, b in enumerate(dynamic_bodies):
            new_position = b.position + b.velocity*time_step + 0.5*total_accel[i]*(time_step**2)
            b.position = new_position
            
            if isinstance(b,Spacecraft): # Checking for crash between ship and planet
                for b_c in Body._instances:
                    if b_c is b:
                        continue
                    if np.linalg.norm(b.position - b_c.position) < b_c.radius:
                        b.is_crashed = True
                        print(f"Crash between {b.name} and {b_c.name}")
        
        new_total_accel = np.zeros((N, 2))    # total acceleration including thrust
        gravity_array = Body.compute_gravity_from_sources_to_targets(dynamic_bodies,sources)
                    
        # compute all new accelerations at t_n+1
        for i, b in enumerate(dynamic_bodies):
            new_grav_accel = gravity_array[i]
            if isinstance(b,Spacecraft):
                new_thrust = b.thrust_ctrl_law(new_grav_accel)
                new_total_accel[i] = new_grav_accel + new_thrust
            else:
                new_total_accel[i] = new_grav_accel 
                
        # compute all new velocities at t_n+1 
        for i, b in enumerate(dynamic_bodies):
            b.velocity = b.velocity + (1/2)*(total_accel[i] + new_total_accel[i])*time_step
            if isinstance(b,Spacecraft):
                b.path = np.append(b.path, [b.position.copy()], axis=0)
            
class Spacecraft(Body):
    _instances = []
    _index_counter = -999
    target_pointer = None

    def __init__(self, name, mass, position, velocity = (0,0), color = 'white', thrust=0.0, orientation=0.0, radius = 1, is_dynamically_updated = True, is_target = False, velocity_vec = False, thrust_vec = False):
        super().__init__(name, mass, position, velocity, color, radius, is_dynamically_updated)
        self.max_thrust = thrust
        self.orientation = orientation  # radians
        self.path = np.array([self.position.copy()])  # Store the path as an array of positions
        self.list_boosters_on = {'up': 0,'down':0, 'left': 0, 'right': 0}
        self.fuel_spent = 0.0
        self.is_target = is_target
        self.navigation_strategy = 'none' # control law
        self.desired_path = None
        
        
        self.ics = (self.position.copy(),self.velocity.copy()) # initial conditions
        self.i_p = self.position.copy() # initial position
        self.i_v = self.velocity.copy() # initial velocity vector


        self.velocity_vec = velocity_vec
        self.thrust_vec = thrust_vec
        self.thrust = np.array([0.0, 0.0])
        self.thrust_mag = np.linalg.norm(self.thrust)
        
        # For debugging and plotting
        
        if self.name == 'target':
            Spacecraft.target_pointer = self
            
            if not self.is_target:
                raise ValueError("Spacecraft named 'target' must have is_target=True")
        
            if any(ship.name == 'target' and ship is not self for ship in Spacecraft._instances):
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
    
    def thrust_ctrl_law(self,accel_from_planets):
        if self.navigation_strategy == 'stay_put':
            ground = self.i_p
            e = self.position - ground
            e_dot = self.velocity - 0
            
            K_p = 0.2
            K_d = 0.3
            
            ship_thrust = -accel_from_planets - K_p * e - K_d * e_dot

        elif self.navigation_strategy == 'thrust_towards_target':
            self.orientation = math.atan2(Spacecraft.target_pointer.position[1] - self.position[1], Spacecraft.target_pointer.position[0] - self.position[0])
            ship_thrust = self.propulsion_acceleration(self.max_thrust, self.orientation)

        elif self.navigation_strategy == 'counteract_gravity':
            pass
            # Fill this in
        elif self.navigation_strategy == 'path-follow':
            pass
            # fill this in
        else:
            ship_thrust = np.array([0.0,0.0])
        thrust_mag = np.linalg.norm(ship_thrust)
        if thrust_mag > self.max_thrust:
            ship_thrust = self.max_thrust * (ship_thrust / thrust_mag)
        return ship_thrust