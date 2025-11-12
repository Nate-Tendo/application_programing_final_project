# test_classes.py
import math
import types
import numpy as np
import pytest

# Import your actual code file
import physics_classes


# -----------------------
# Helpers & small fakes
# -----------------------
class SimpleBody:
    """Minimal object with position and mass used by PhysicsEngine tests."""
    def __init__(self, pos, mass):
        self.position = np.array(pos, dtype=float)
        self.mass = mass


class FakeShip:
    """Minimal ship-like object for integrate_rk4 tests."""
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass


# -----------------------
# Tests for Bounds dataclass
# -----------------------
def test_bounds_dataclass_fields():
    b = physics_classes.Bounds(x_min=-10.0, x_max=10.0, y_min=-5.0, y_max=5.0)
    assert b.x_min == -10.0
    assert b.x_max == 10.0
    assert b.y_min == -5.0
    assert b.y_max == 5.0


# -----------------------
# CelestialBody tests
# -----------------------
def test_gravitational_acceleration_from_body_monopole():
    # Ensure module has GRAVITY_CONSTANT available for the method
    orig_G = getattr(physics_classes, "GRAVITY_CONSTANT", None)
    physics_classes.GRAVITY_CONSTANT = 2.0  # set test G

    a = physics_classes.CelestialBody("A", mass=5.0, position=[0.0, 0.0], radius=1.0)
    b = physics_classes.CelestialBody("B", mass=10.0, position=[3.0, 4.0], radius=1.0)

    # vector from a to b is [3,4], magnitude 5. Expected acceleration:
    expected = 20.0 * np.array([3.0, 4.0]) / 125.0
    res = a.gravitational_acceleration_from_body(b)
    assert np.allclose(res, expected)

    # restore original
    if orig_G is None:
        delattr(physics_classes, "GRAVITY_CONSTANT")
    else:
        physics_classes.GRAVITY_CONSTANT = orig_G


# -----------------------
# DynamicBody tests
# -----------------------
def test_propulsion_acceleration_direction_magnitude():
    db = physics_classes.DynamicBody("ship", mass=2.0, position=[0, 0], velocity=[0, 0], radius=1.0)

    res = db.propulsion_acceleration(10.0, 0.0)
    assert pytest.approx(res[0]) == 10.0 / 2.0
    assert pytest.approx(res[1]) == 0.0

    res2 = db.propulsion_acceleration(10.0, math.pi / 2.0)
    assert pytest.approx(res2[0], abs=1e-9) == 0.0
    assert pytest.approx(res2[1]) == 10.0 / 2.0


def test_update_trail_trimming_behavior():
    db = physics_classes.DynamicBody("ship", mass=1.0, position=[0, 0], velocity=[0, 0], radius=1.0)
    db.trail = [np.array([i, i], dtype=float) for i in range(2100)]
    db.update_trail()
    assert len(db.trail) == 2100
    assert np.allclose(db.trail[-1], db.position)


def test_compute_total_current_force_with_boosters_and_stubbed_gravity(monkeypatch):
    db = physics_classes.DynamicBody("ship", mass=2.0, position=[0.0, 0.0], velocity=[0.0, 0.0], radius=1.0)

    def fake_grav(self, other):
        return np.array([1.0, 1.0])
    monkeypatch.setattr(physics_classes.DynamicBody, "gravitational_acceleration_from", fake_grav, raising=False)

    monkeypatch.setattr(physics_classes, "Spacecraft", type("Spacecraft", (), {}), raising=False)
    fake_Body = types.SimpleNamespace(_instances=[object(), object()])
    monkeypatch.setattr(physics_classes, "Body", fake_Body, raising=False)

    def fake_prop(self, thrust_magnitude, thrust_direction):
        return np.array([thrust_magnitude, thrust_direction])
    monkeypatch.setattr(physics_classes.DynamicBody, "propulsion_acceleration", fake_prop, raising=False)

    db.list_boosters_on['up'] = 1
    db.list_boosters_on['right'] = 1

    total = db.compute_total_current_force()

    expected_boosters = np.array([np.deg2rad(90), 0.0])
    expected_total = np.array([2.0, 2.0]) + expected_boosters
    assert np.allclose(total, expected_total)
    assert db.list_boosters_on['up'] == 0
    assert db.list_boosters_on['right'] == 0


# -----------------------
# PhysicsEngine tests
# -----------------------
def test_physicsengine_gravitational_acceleration_zero_and_nonzero(monkeypatch):
    monkeypatch.setattr(physics_classes.PhysicsEngine, "G", 3.0, raising=False)

    b1 = SimpleBody([1.0, 0.0], mass=2.0)
    b2 = SimpleBody([0.0, 2.0], mass=1.0)

    res = physics_classes.PhysicsEngine.gravitational_acceleration(np.array([0.0, 0.0]), [b1, b2])

    def single(b):
        rvec = b.position - np.array([0.0, 0.0])
        r = np.linalg.norm(rvec)
        return 3.0 * b.mass * rvec / (r ** 3)
    expected = single(b1) + single(b2)
    assert np.allclose(res, expected)


def test_integrate_rk4_conserves_simple_behavior(monkeypatch):
    ship = FakeShip(position=[0.0, 0.0], velocity=[1.0, 0.0], mass=2.0)
    bodies = []
    dt = 0.1
    thrust = np.array([0.0, 0.0])
    monkeypatch.setattr(physics_classes.PhysicsEngine, "G", 1.0, raising=False)

    pos0 = ship.position.copy()
    vel0 = ship.velocity.copy()
    physics_classes.PhysicsEngine.integrate_rk4(ship, bodies, dt, thrust=thrust)

    assert ship.position.shape == pos0.shape
    assert ship.position[0] > pos0[0]
    assert np.isfinite(ship.velocity).all()


# -----------------------
# TrajectoryPredictor tests
# -----------------------
def test_trajectory_predictor_creates_prediction_with_monkeypatched_dynamicbody(monkeypatch):
    fake_univ = types.SimpleNamespace(bodies=[])
    ship = types.SimpleNamespace(name="S", position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0)

    class FakeDyn:
        def __init__(self, name, pos, vel, mass):
            self.name = name
            self.position = np.array(pos, dtype=float)
            self.velocity = np.array(vel, dtype=float)
            self.mass = mass

    monkeypatch.setattr(physics_classes, "DynamicBody", FakeDyn, raising=False)
    predictor = physics_classes.TrajectoryPredictor(fake_univ)

    goal = np.array([1e8, 0.0])
    predicted = predictor.predict(ship, goal, dt=0.1, steps=3, thrust_mag=0.0)
    assert isinstance(predicted, np.ndarray)
    assert predicted.ndim == 2
    assert predicted.shape[0] >= 1


# -----------------------
# Sanity: existence
# -----------------------
def test_smoke_imports_and_key_attributes():
    assert hasattr(physics_classes, "CelestialBody")
    assert hasattr(physics_classes, "DynamicBody")
    assert hasattr(physics_classes, "PhysicsEngine")
    assert hasattr(physics_classes, "TrajectoryPredictor")
