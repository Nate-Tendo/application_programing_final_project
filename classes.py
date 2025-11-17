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
        self.path = np.array([self.position.copy()])
        
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
    
    # Computes the gravitational acceleration from all other bodies on one body
    def g_from_bodies(self):
        accel_vec = sum((self.gravitational_acceleration_from(body)
                    for body in Body._instances if (body is not self) 
                    and (not isinstance(body, Spacecraft))), np.array([0,0]))
        return accel_vec
    
    # Velocity Verlet Numerical Integration -  computes each step for all bodies to avoid violating conservation of momentum   
    # @staticmethod
    def timestep(self, time_step = 1):
        dynamic_bodies = [body for body in Body._instances if body.is_dynamically_updated == True]
        
        # Using dictionaries to index off body name to avoid array index errors

        grav_accel = {}
        total_accel = {}
        thrusts = {}
        
        # compute accelerations at t_n
        for b in dynamic_bodies:
            grav_accel[b] = b.g_from_bodies()
            b.grav_accel = grav_accel[b]
            if isinstance(b,Spacecraft):
                thrusts[b] = b.thrust_ctrl_law(grav_accel[b])
                b.thrust = thrusts[b]
                b.thrust_mag = np.linalg.norm(thrusts[b])
                b.fuel_spent += (np.linalg.norm(thrusts[b]) * b.mass) * time_step
                total_accel[b] = grav_accel[b] + thrusts[b]
            else:
                total_accel[b] = grav_accel[b]
        
        # compute all new positions at t_n+1
        for b in dynamic_bodies:
            new_position = b.position + b.velocity*time_step + 0.5*total_accel[b]*(time_step**2)
            
            for b_c in Body._instances:
                if b_c is b:
                    continue
                elif segment_circle_intersect(b.position, new_position, b_c.position, b_c.radius): #Remember that position.v is the position vector!
                    b.is_crashed = True
                    b.is_dynamically_updated = False
                    b_c.is_crashed = True
                    b_c.is_dynamically_updated = False
                    print(f"Crash between {b.name} and {b_c.name}")
                        
            # If not crashed, then update position                
            b.position = new_position
        
        new_total_accel = {}
                    
        # compute all new accelerations a t_n+1
        for b in dynamic_bodies:
            new_grav_accel = b.g_from_bodies()
            if isinstance(b,Spacecraft):
                new_thrust = b.thrust_ctrl_law(new_grav_accel)
                new_total_accel[b] = new_grav_accel + new_thrust
            else:
                new_total_accel[b] = new_grav_accel 
                
        # compute all new velocities at t_n+1 
        for b in dynamic_bodies:
            b.velocity = b.velocity + (1/2)*(total_accel[b] + new_total_accel[b])*time_step
            b.path = np.append(b.path, [b.position.copy()], axis=0)
            
class Spacecraft(Body):
    _instances = []
    _index_counter = -999

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
        # Direction PID
        self.mpid_integral_dir = 0.0
        self.mpid_last_error_dir = 0.0
        # Magnitude PID
        self.mpid_integral_mag = 0.0
        self.mpid_last_error_mag = 0.0

        # momentum setpoint (tune this)
        self.mpid_setpoint = 2000.0   # example

        
        
        self.ics = (self.position.copy(),self.velocity.copy()) # initial conditions
        self.i_p = self.position.copy() # initial position
        self.i_v = self.velocity.copy() # initial velocity vector


        self.velocity_vec = velocity_vec
        self.thrust_vec = thrust_vec
        self.thrust = np.array([0.0, 0.0])
        self.thrust_mag = np.linalg.norm(self.thrust)
        
        # For debugging and plotting

        if self.name == 'target' and not self.is_target:
            raise ValueError("Spacecraft named 'target' must have is_target=True")
        
        if self.is_target and any(ship.name == 'target' and ship is not self for ship in Spacecraft._instances):
            raise ValueError("There can only be one target spacecraft")

        Spacecraft._instances.append(self)
        if len(Spacecraft._instances) == 0:
            Spacecraft._index_counter = 0
        else:
            Spacecraft._index_counter += 1
            
    def set_nav_strat(self, nav_strat = 'none', path_follow = None):
        self.path_follow = path_follow
        self.nav_strat = nav_strat
        if path_follow:    
            if nav_strat == 'line_follow':
                self.path_start = np.array(self.position)
                self.path_end = np.array(path_follow)
                self.path_vec = self.path_start-self.path_end
                self.path_len = np.linalg.norm(self.path_vec)
                self.path_unitvec = self.path_vec/self.path_len
            if nav_strat == 'path_follow':
                pass

    # TDOO: Update propulsion to take in orietnation relative to spacecraft and translate to the environment
    def propulsion_acceleration(self,thrust_magnitude,thrust_direction):
        ax = thrust_magnitude * np.cos(thrust_direction) / self.mass
        ay = thrust_magnitude * np.sin(thrust_direction) / self.mass
        return np.array([ax,ay])
    
    def thrust_ctrl_law(self,g): # Determines the control strategy the ship will use
        match self.nav_strat:
            case 'stay_put': # Stay put in the starting position
                ground = np.array(self.i_p)
                e = self.position - ground
                e_dot = self.velocity - 0
                
                K_p = 0.3
                K_d = 0.2
                
                ship_thrust = -g - K_p * e - K_d * e_dot
            case 'thrust_towards_target':
                ship_thrust = self.propulsion_acceleration(self.max_thrust, self.orientation)
            case 'counteract_gravity':
                pass
                # Fill this in
            case 'line_follow': # Follow a linear path from start to end then stay put 
                # Functionality must include both
                # - Ability to get back onto the path when it moves off
                # - Ability to slow down as it approaches the end point
                # - Ability to use gravity to move us towards the end to reduce thrust usage
            
                # Control components: 1) cancel tangent gravity, 2) move along path, 3)
            
                n = self.path_unitvec
                g_tan = np.dot(g,n)*n
                g_norm = g - g_tan
                
                d_vec = self.position - self.path_start
                d = np.linalg.norm(d_vec)
                
                v_tan = np.dot(self.velocity,n)*n
                v_norm = self.velocity - v_tan
                
                # Define coordinate as projection of position vector onto path unit vector
                s = np.dot(d_vec, n)
                
                # Bottoms out the coordinate if it is less than 0 or more than L
                s_coord = np.clip(s,0,self.path_len)
                    
                p_near = s - s_coord
                
                e = self.position-self.path_end
                e_tan = np.dot(e,n)*n
                e_norm = e-e_tan
                
                # if v_tan !=0
                                               
                if np.linalg.norm(e_tan)<10: # once we are near, stay put
                    ground = self.path_end
                    e = self.position - ground
                    e_dot = self.velocity - 0
                    
                    K_p = 0.3
                    K_d = 0.2
                    
                    ship_thrust = -g - K_p * e - K_d * e_dot
                else:
                    K_p = 0.1
                    K_d = 0.1
                    additional_thrust = -K_p*e_norm               
                    
                    # Removes the normal component of gravity so gravity can assist us along the path
                    ship_thrust = -g_norm + additional_thrust
            case 'path_follow': # Follow an arbitrary spline from start to end then stay put
                
                
                thrust = 1
                # fill this in
            case 'potential_field':
                k_att = 0.90      # attractive gain (accel per unit distance)
                k_rep = 1.0   # repulsive gain (scale of repulsion)
                d0 = 1000.0        # influence radius of obstacles (any body closer than d0 repels)
                # Add mass scaling

                target = None
                # finding the target
                for ship in Spacecraft._instances:
                    if ship.is_target and ship is not self:
                        target = ship
                        break

                if target is None:
                    desired_accel = np.array([0.0, 0.0])
                    print('No Target Found')
                else:
                    # Attractive acceleration
                    pos = self.position
                    dir_to_goal = target.position - pos
                    dist_to_goal = np.linalg.norm(dir_to_goal)
                    att_accel = k_att * dir_to_goal  # direction * distance * k_att

                    # Repulsive acceleration 
                    rep_accel = np.array([0.0, 0.0])
                    for body in Body._instances:
                        if body is self:
                            continue
                        # treat target as not an obstacle for repulsion (optional)
                        if isinstance(body, Spacecraft) and body.is_target:
                            continue
                        vec = pos - body.position
                        dist = np.linalg.norm(vec)
                        if dist < (body.radius * 10): # If overlapping, create a large immediate repulsion
                            repulse = (vec + 1e-3) * k_rep * 10.0
                            rep_accel += repulse
                            print('ah')
                            continue
                        if dist <= d0:
                            # unit vector away from obstacle
                            u = vec / dist
                            # magnitude: k_rep * (1/d - 1/d0)
                            mag = k_rep * (1.0/dist - 1.0/d0)
                            rep_accel += u * mag
                            print('oh')

                    # Desired guidance acceleration (combine)
                    desired_accel = att_accel + rep_accel

                    v = self.velocity
                    v_mag = np.linalg.norm(v)
                    if v_mag < 1e-8:
                        v_dir = np.array([0.0, 0.0])
                    else:
                        v_dir = v / v_mag

                    # Normalize desired direction
                    da_mag = np.linalg.norm(desired_accel)
                    if da_mag < 1e-8:
                        desired_dir = v_dir
                    else:
                        desired_dir = desired_accel / da_mag


                    # MOMENTUM–DIRECTION PID (STEERING)
                    # Signed angular error between desired_dir and v_dir
                    angle_error = np.arctan2(
                        v_dir[0] * desired_dir[1] - v_dir[1] * desired_dir[0],
                        np.dot(v_dir, desired_dir)
                    )

                    Kp_turn = 2.0
                    Ki_turn = 0.05
                    Kd_turn = 0.5

                    self.mpid_integral_dir += angle_error

                    d_angle = angle_error - self.mpid_last_error_dir
                    self.mpid_last_error_dir = angle_error

                    # Turn command (scalar)
                    turn_cmd = (
                        Kp_turn * angle_error +
                        Ki_turn * self.mpid_integral_dir +
                        Kd_turn * d_angle
                    )

                    # Perpendicular direction (90° rotated)
                    if v_mag > 1e-6:
                        perp = np.array([-v_dir[1], v_dir[0]])
                        a_turn = turn_cmd * perp
                    else:
                        a_turn = np.array([0.0, 0.0])

                    # 3. MOMENTUM–MAGNITUDE PID (SPEED LIMITING)
                    momentum = self.mass * v_mag
                    error_m = self.mpid_setpoint - momentum   # positive = too slow, negative = too fast

                    Kp_spd = 0.01
                    Ki_spd = 0.0001
                    Kd_spd = 0.005

                    self.mpid_integral_mag += error_m

                    d_error_m = error_m - self.mpid_last_error_mag
                    self.mpid_last_error_mag = error_m

                    # scalar output
                    spd_cmd = (
                        Kp_spd * error_m +
                        Ki_spd * self.mpid_integral_mag +
                        Kd_spd * d_error_m
                    )

                    # speed control acceleration:
                    #   - when momentum > setpoint → spd_cmd < 0 → brake opposite v_dir
                    #   - when momentum < setpoint → spd_cmd > 0 → allow some boost along v_dir
                    if v_mag > 1e-6:
                        a_speed = v_dir * spd_cmd
                    else:
                        a_speed = np.array([0.0, 0.0])

                    final_desired_accel = desired_accel + a_turn + a_speed

                    # Convert to thrust
                    thrust_vec = final_desired_accel * self.mass
                    thrust_mag = np.linalg.norm(thrust_vec)

                    # clamp
                    if thrust_mag > self.max_thrust > 0:
                        thrust_vec = thrust_vec / thrust_mag * self.max_thrust

                    ship_thrust = thrust_vec

                    print(ship_thrust)
            case __:
                ship_thrust = np.array([0.0,0.0])
        thrust_mag = np.linalg.norm(ship_thrust)
        if thrust_mag > self.max_thrust and thrust_mag > 0:
                ship_thrust = ship_thrust / thrust_mag * self.max_thrust
        return ship_thrust