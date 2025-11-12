# THIS TEST SCRIPT WAS GENERATED WITH HELP FROM CHAT GPT

import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.ticker import MultipleLocator
from physics_classes import Vector2, Body, Spacecraft

def plot_universe(ax, window=100):
    colors_bodies = [body.color for body in Body._instances]
    for body in Body._instances:
        circle = plt.Circle((body.position.x, body.position.y), body.radius, color=body.color)
        ax.add_patch(circle)
    ax.set_facecolor('black')
    ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(window / 10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=[-window, window], ylim=[-window, window])
    plt.tight_layout(pad=0.5)

def cleanup_losing_ships(winner):
    """Remove all ships except the winning one from the global Body list."""
    survivors = []
    for b in Body._instances:
        # Keep stars and the winner
        if isinstance(b, Spacecraft):
            if b.name == winner.name:
                survivors.append(b)
        else:
            survivors.append(b)
    Body._instances = survivors

def simulate_path(ship, dt, t_span):
    """Simulate ship for t_span seconds; return (path, min_dist_to_target, crashed)."""
    path_x, path_y = [], []
    temp_ship = copy.deepcopy(ship)
    crashed = False

    for _ in np.arange(0, t_span, dt):
        crashed = temp_ship.step_forward_dt(time_step=dt)
        path_x.append(temp_ship.position.x)
        path_y.append(temp_ship.position.y)
        if crashed:
            break

    return (path_x, path_y, temp_ship.position, crashed)



Body._instances.clear()

star1 = Body(name='star1', mass=2000, position=Vector2(0, 0), velocity=Vector2(0, 0),
             color='blue', radius=50)

base_ship = Spacecraft(name='ShipBase', mass=10, position=Vector2(0, -500),
                       velocity=Vector2(0, 0), thrust=2.0, color='white', radius=10)

target = Spacecraft(name='target', mass=0, position=Vector2(-500, 500),
                    velocity=Vector2(0, 0), color='purple', radius=20)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
plot_universe(ax, window=1000)

dt = 1.0
t_span = 150.0
tolerance = 5.0  # distance to target to stop
max_iterations = 1000

current_ship = copy.deepcopy(base_ship)

for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}")

    # Create 4 candidate directions
    candidates = []
    for direction, color in zip(
        ['up', 'down', 'left', 'right'],
        ['red', 'cyan', 'yellow', 'lime']
    ):
        ship = copy.deepcopy(current_ship)
        ship.name = f"{iteration} Ship "
        ship.list_boosters_on = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        ship.list_boosters_on[direction] = 1
        ship.color = color
        candidates.append(ship)

    # Simulate each candidate
    best_dist = float('inf')
    best_ship = None
    best_path = None

    for ship in candidates:
        path_x, path_y, final_pos, crashed = simulate_path(ship, dt, t_span)
        dist = (final_pos - target.position).magnitude()
        ax.plot(path_x, path_y, '--', color=ship.color, alpha=0.7)
        # print(f"  {ship.name}: dist={dist:.1f}, crashed={crashed}")
        if not crashed and dist < best_dist:
            best_dist = dist
            best_ship = ship
            best_path = (path_x, path_y)

    if best_ship is None:
        print("All candidates crashed! Ending simulation.")
        break
    
    print(f"  {ship.name}: dist={dist:.1f}, booster={direction}")

    cleanup_losing_ships(best_ship)
    # Move to best path’s endpoint
    current_ship = best_ship
    ax.plot(best_path[0], best_path[1], '-', color='white', linewidth=2)

    # Stop if close to target
    if best_dist < tolerance:
        print(f"Target reached within {best_dist:.1f} units.")
        break

    # # Optional: adapt step size to get finer as we get close
    # t_span *= 0.8
    # dt *= 0.8

# Draw target marker
ax.scatter(target.position.x, target.position.y, s=200, color='magenta', marker='X')
plt.show()
