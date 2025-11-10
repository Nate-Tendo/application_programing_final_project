import numpy as np
import random

class CelestialBody:
    def __init__(self, name, mass, position, radius=1e7, color='yellow'):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.color = color


class Spaceship:
    def __init__(self, name, position, velocity=[0, 0], mass=1000, color='white'):
        self.name = name
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.color = color
        self.trail = [self.position.copy()]
        self.thrust_on = False
        self.orientation = 0.0  # radians

    def update_trail(self):
        if len(self.trail) > 2000:
            self.trail.pop(0)
        self.trail.append(self.position.copy())


class PhysicsEngine:
    G = 6.67430e-11

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
    def __init__(self, num_bodies=4, bounds=(-1e9, 1e9, -1e9, 1e9)):
        self.bounds = bounds
        self.bodies = []
        self.ships = []
        self.generate_bodies(num_bodies)

    def generate_bodies(self, n):
        xmin, xmax, ymin, ymax = self.bounds
        for i in range(n):
            pos = [random.uniform(xmin, xmax), random.uniform(ymin, ymax)]
            mass = random.uniform(1e23, 1e26)
            radius = random.uniform(5e6, 2e7)
            color = random.choice(['yellow', 'orange', 'blue', 'red'])
            self.bodies.append(CelestialBody(f"Body{i}", mass, pos, radius, color))

    def add_ship(self, ship):
        self.ships.append(ship)


class TrajectoryPredictor:
    """Predicts trajectory under gravity + thrust guidance toward goal."""
    def __init__(self, universe):
        self.universe = universe

    def predict(self, ship, goal, dt=1000, steps=20000, thrust_mag=5e-3):
        s = Spaceship(ship.name + "_pred", ship.position.copy(), ship.velocity.copy(), ship.mass)
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
