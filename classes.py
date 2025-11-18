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
    
    def __init__(self, name: str, mass: float, position: tuple, velocity = (0,0), color = 'blue', 
                 radius = 10, is_dynamically_updated = True, velocity_vec = False):
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

        # Move these here so they can be accessed for all bodies
        self.ics = (self.position.copy(),self.velocity.copy()) # initial conditions
        self.i_p = self.position.copy() # initial position
        self.i_v = self.velocity.copy() # initial velocity vector
        self.i_dynamic_state = self.is_dynamically_updated
        self.i_path = self.path.copy() # initial path


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
    @staticmethod
    def timestep(time_step = 1):
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

    def __init__(self, name, mass, position, velocity = (0,0), color = 'white', thrust=0.0, 
                 orientation=0.0, radius = 1, is_dynamically_updated = True, is_target = False,
                 velocity_vec = False, thrust_vec = False, acc_vec = False, vector_field = True):
        super().__init__(name, mass, position, velocity, color, radius, is_dynamically_updated)
        self.max_thrust = thrust
        self.orientation = orientation  # radians
        self.path = np.array([self.position.copy()])  # Store the path as an array of positions
        self.list_boosters_on = {'up': 0,'down':0, 'left': 0, 'right': 0}
        self.fuel_spent = 0.0
        self.is_target = is_target
        self.navigation_strategy = 'none' # control law
        self.desired_path = None
        self.accel = np.array([0.0,0.0]) #for debugging
        # For smoothing commanded acceleration (potential field modes)
        self.prev_a_cmd = np.array([0.0, 0.0], dtype=float)
        self.nav_strat = 'none' # control law
        self.path_follow = None
        # Direction PID
        self.mpid_integral_dir = 0.0
        self.mpid_last_error_dir = 0.0
        # Magnitude PID
        self.mpid_integral_mag = 0.0
        self.mpid_last_error_mag = 0.0

        # momentum setpoint (tune this)
        self.mpid_setpoint = 20.0   # example

        self.velocity_vec = velocity_vec
        self.thrust_vec = thrust_vec
        self.thrust = np.array([0.0, 0.0])
        self.thrust_mag = np.linalg.norm(self.thrust)

        # This is semi-cheating, but it'll work for now
        self.plot_vectorfield = vector_field
        self.path_visible = True
        self.plot_potentialfield = False
        self.planet_path_visible = False
        
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
    
    def _find_target(self):
        for ship in Spacecraft._instances:
            if ship.is_target and ship is not self:
                return ship
        return None

    def _repulsive_accel(self, pos, repulsion_factor=10.0, k_rep=1.5e5):
        """Shared obstacle repulsion (returns acceleration)."""
        a_rep = np.zeros(2, dtype=float)
        obstacles = [b for b in Body._instances if not isinstance(b, Spacecraft)]
        for body in obstacles:
            vec = pos - body.position
            d = np.linalg.norm(vec)
            if d < 1e-9:
                continue

            safe_zone = body.radius * repulsion_factor

            # If we somehow get inside the body radius, huge push out
            if d < body.radius:
                print('bad')
                a_rep += (vec / d) * 5e3
                continue

            if d < safe_zone:
                # classic repulsive field, scaled by distance and safe_zone
                print('pushing')
                a_rep += k_rep * (1.0 / d - 1.0 / safe_zone) * (1.0 / (d**2)) * (vec / d)
        return a_rep

    def thrust_ctrl_law(self, g):
        """
        Returns thrust ACCELERATION vector in world frame.

        g: gravitational acceleration on this body from other bodies (np.array shape (2,))
        """
        match self.nav_strat:
            # -------------------------------------------------
            # 1. existing modes (unchanged)
            # -------------------------------------------------
            case 'stay_put':
                ground = self.i_p
                e = self.position - ground
                e_dot = self.velocity - 0

                K_p = 0.2
                K_d = 0.3
                ship_thrust = -g - K_p * e - K_d * e_dot

            case 'thrust_towards_target':
                ship_thrust = self.propulsion_acceleration(self.max_thrust, self.orientation)

            case 'line_follow':
                n = self.path_unitvec
                g_tan = np.dot(g, n) * n
                g_norm = g - g_tan

                d_vec = self.position - self.path_start
                d = np.linalg.norm(d_vec)

                v_tan = np.dot(self.velocity, n) * n
                v_norm = self.velocity - v_tan

                e = self.path_end - self.position
                e_tan = np.dot(e, n) * n
                e_norm = e - e_tan

                if np.linalg.norm(e_tan) < 10:
                    ground = self.path_end
                    e = self.position - ground
                    e_dot = self.velocity - 0
                    K_p = 0.1
                    K_d = 0.3
                    ship_thrust = -g - K_p * e - K_d * e_dot - K_d * v_norm
                else:
                    K_p_norm = 0.02
                    K_d_norm = 0.25
                    K_d_tan = 0.002
                    additional_thrust = -K_p_norm * e_norm + K_d_norm * v_norm + K_d_tan * v_tan
                    ship_thrust = -g_norm - additional_thrust

            # -------------------------------------------------
            # 2. IMPROVED POTENTIAL FIELD (with tangential braking)
            # -------------------------------------------------
            case 'potential_field':
                # Potential-field guidance with chase + rendezvous modes
                target = self._find_target()
                if target is None:
                    # no target: just hover by cancelling gravity and damping
                    k_damp_hover = 0.08
                    ship_thrust = -g - k_damp_hover * self.velocity
                else:
                    pos = self.position
                    vel = self.velocity
                    tgt_pos = target.position
                    tgt_vel = target.velocity

                    # -------------------------
                    # Params to tune
                    # -------------------------
                    k_att_far   = 8e-4     # attractive gain when far
                    k_att_near  = 1.2e-3   # attractive gain when near (rendezvous PD)
                    k_damp_far  = 0.03     # small global damping when far
                    k_damp_near = 0.10     # damping on *relative* velocity when near

                    chase_radius     = 400.0   # outside this → chase mode #TODO: Could likely scale these based on initial distance to target
                    rendezvous_radius = 200.0  # inside this → fully rendezvous

                    repulsion_factor = 8.0
                    k_rep            = 8e7
                    a_rep_max        = 5    # cap on repulsive accel
                    alpha_smooth     = 0.25    # accel smoothing  (0<alpha<=1)

                    # -------------------------
                    # Common terms
                    # -------------------------
                    dir_to_goal = tgt_pos - pos          # vector TO target
                    dist_to_goal = np.linalg.norm(dir_to_goal)
                    if dist_to_goal < 1e-6:
                        dir_to_goal = np.zeros(2)
                        dist_to_goal = 0.0

                    # repulsive acceleration (from planets etc.)
                    a_rep = self._repulsive_accel(
                        pos,
                        repulsion_factor=repulsion_factor,
                        k_rep=k_rep,
                    )
                    rep_mag = np.linalg.norm(a_rep)
                    if rep_mag > a_rep_max and rep_mag > 0:
                        a_rep = a_rep / rep_mag * a_rep_max

                    # -------------------------
                    # Mode 1: CHASE (far away)
                    #   - run toward target
                    #   - only light global damping
                    # -------------------------
                    if dist_to_goal > chase_radius:
                        a_att = k_att_far * dir_to_goal
                        a_damp = -k_damp_far * vel               # damp absolute velocity a bit
                        a_total_des = a_att + a_rep + a_damp

                    # -------------------------
                    # Mode 2: RENDEZVOUS (near target)
                    #   - PD on relative position and velocity
                    #   - matches target motion instead of stopping in world frame
                    # -------------------------
                    else:
                        # position error (self relative to target)
                        e = pos - tgt_pos          # want e → 0
                        # relative velocity
                        v_rel = vel - tgt_vel      # want v_rel → 0

                        # blend gains depending on distance (optional smooth transition)
                        # here we just use "near" gains inside chase_radius
                        k_p = k_att_near
                        k_v = k_damp_near

                        # desired TOTAL accel (including gravity) in relative coordinates
                        a_rel_des = -k_p * e - k_v * v_rel

                        # plus obstacle repulsion in world frame
                        a_total_des = a_rel_des + a_rep

                    # -------------------------
                    # Smooth the command a bit
                    # -------------------------
                    a_cmd = (1.0 - alpha_smooth) * self.prev_a_cmd + alpha_smooth * a_total_des
                    self.prev_a_cmd = a_cmd.copy()

                    # -------------------------
                    # Convert desired total accel to thrust accel
                    #
                    # timestep() does: total = g + thrust
                    # We want total ≈ a_cmd  ⇒ thrust = a_cmd - g
                    # -------------------------
                    ship_thrust = a_cmd - g


            # -------------------------------------------------
            # 3. LYAPUNOV-STABLE PD CONTROLLER (with gravity comp.)
            #    V = 0.5||e||^2 + 0.5||v||^2
            # -------------------------------------------------
            case 'lyapunov_pd':
                target = self._find_target()
                if target is None:
                    ship_thrust = -g
                else:
                    e = self.position - target.position          # position error
                    v = self.velocity - target.velocity          # velocity error

                    k_p = 0.001
                    k_v = 0.05

                    # desired total acceleration (including gravity & thrust)
                    a_total_des = -k_p * e - k_v * v + self._repulsive_accel(self.position)

                    # Body.timestep will do total = g + thrust,
                    # so thrust accel must be a_total_des - g
                    ship_thrust = a_total_des - g

            # -------------------------------------------------
            # 4. LYAPUNOV + NONLINEAR DAMPING
            #    adds term -k_v2 ||v|| v to kill hunting
            # -------------------------------------------------
            case 'lyapunov_nonlinear':
                target = self._find_target()
                if target is None:
                    ship_thrust = -g
                else:
                    e = self.position - target.position
                    v = self.velocity - target.velocity

                    k_p = 0.001
                    k_v_lin = 0.03
                    k_v_nl = 0.01   # nonlinear damping gain

                    vmag = np.linalg.norm(v)
                    if vmag > 1e-6:
                        v_hat = v / vmag
                        a_damp_nl = -k_v_nl * (vmag**2) * v_hat
                    else:
                        a_damp_nl = np.zeros(2)

                    a_total_des = -k_p * e - k_v_lin * v + a_damp_nl + self._repulsive_accel(self.position)

                    ship_thrust = a_total_des - g

            # -------------------------------------------------
            # 5. NAVIGATION FUNCTION CONTROLLER
            #    Global minimum at goal, obstacles as “maxima”
            # -------------------------------------------------
            case 'nav_function':
                target = self._find_target()
                if target is None:
                    ship_thrust = -g
                else:
                    q = self.position
                    qd = target.position

                    e = q - qd
                    num = np.dot(e, e)

                    # obstacle β(q)
                    obstacles = [b for b in Body._instances if not isinstance(b, Spacecraft)]
                    betas = []
                    diffs = []
                    radii = []
                    repulsion_factor = 8.0

                    for body in obstacles:
                        diff = q - body.position
                        d2 = np.dot(diff, diff)
                        R = body.radius * repulsion_factor
                        bi = d2 - R**2
                        betas.append(bi)
                        diffs.append(diff)
                        radii.append(R)

                    if betas:
                        beta = 1.0
                        for bi in betas:
                            beta *= bi

                        # grad β(q)
                        grad_beta = np.zeros(2)
                        for i in range(len(betas)):
                            bi = betas[i]
                            diff_i = diffs[i]
                            grad_bi = 2.0 * diff_i

                            prod_others = 1.0
                            for j in range(len(betas)):
                                if j == i:
                                    continue
                                prod_others *= betas[j]
                            grad_beta += grad_bi * prod_others
                    else:
                        beta = 1.0
                        grad_beta = np.zeros(2)

                    k_nav = 1.0
                    den = num + k_nav * beta
                    if den < 1e-6:
                        ship_thrust = -g
                    else:
                        grad_num = 2.0 * e
                        grad_den = grad_num + k_nav * grad_beta
                        grad_phi = (grad_num * den - num * grad_den) / (den**2)

                        k_phi = 0.5
                        a_total_des = -k_phi * grad_phi   # gradient descent on nav function

                        ship_thrust = a_total_des - g

            # ----------------------------------------------------------
            # 6. Chase -- User-Controlled via Boosters in Universe Frame
            # ----------------------------------------------------------

            case 'chase':
                ## VERY SIMPLE FORCE-BOOSTING SCHEME. Will definitely need to update and probably translate into an appropriate frame
                force_boosters = np.array([0.0,0.0])
            
                if self.list_boosters_on['up'] == 1:
                    
                    force_boosters += self.propulsion_acceleration(self.max_thrust, np.deg2rad(90))
                    self.list_boosters_on['up'] = 0
                    print('up')

                if self.list_boosters_on['down'] == 1:
                    
                    force_boosters += self.propulsion_acceleration(self.max_thrust, np.deg2rad(-90))
                    self.list_boosters_on['down'] = 0
                    print('down')
                    
                if self.list_boosters_on['left'] == 1:
                    
                    force_boosters += self.propulsion_acceleration(self.max_thrust, np.deg2rad(180))
                    self.list_boosters_on['left'] = 0
                    print('left')
                    
                if self.list_boosters_on['right'] == 1:
                    
                    force_boosters += self.propulsion_acceleration(self.max_thrust, np.deg2rad(0))
                    self.list_boosters_on['right'] = 0
                    print('right')

                ship_thrust = force_boosters


            # -------------------------------------------------
            # 7. default
            # -------------------------------------------------
            case __:
                ship_thrust = np.array([0.0,0.0])

        # ------------ GLOBAL ACCELERATION LIMIT -------------
        a = ship_thrust
        a_mag = np.linalg.norm(a)
        if self.max_thrust > 0:
            a_max = self.max_thrust / self.mass
            if a_mag > a_max and a_mag > 0:
                a = a / a_mag * a_max
        return a