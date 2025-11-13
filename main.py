import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import math
from classes import Body, Spacecraft, GRAVITY_CONSTANT, EPSILON_GRAVITY
from solar_system_config import initialize_universe

def compute_line_gravity_cost(x_coords: np.ndarray, y_coords: np.ndarray):
    """
    Compute the gravitational 'cost' along a line defined by x,y coordinates.

    Parameters
    ----------
    x_coords, y_coords : array-like
        Arrays (or lists) of x and y coordinates along the line. Must have the same length.

    Returns
    -------
    total_vector : np.array
        Sum of all equal-and-opposite gravitational vectors (represents total imbalance).
    line_vectors : list of np.array
        The opposite-force vectors at each line point.
    total_magnitude : float
        The magnitude of the summed total_vector (scalar cost).
    """

    assert len(x_coords) == len(y_coords), "x_coords and y_coords must be the same length."

    total_vector = np.array((0.0, 0.0),dtype=float)
    line_vectors = []

    for x, y in zip(x_coords, y_coords):
        point = np.array((x, y),dtype=float)
        total_gravity = np.array((0.0, 0.0),dtype=float)
        for body in Body._instances:
            direction = body.position - point
            distance = np.linalg.norm(direction)
            if distance < EPSILON_GRAVITY:
                continue
            g_force_mag = GRAVITY_CONSTANT * body.mass / (distance**2 + EPSILON_GRAVITY**2)
            total_gravity += (direction / distance) * g_force_mag
        # Equal and opposite (to make net zero)
        opposite_force = total_gravity * -1
        line_vectors.append(opposite_force)
        total_vector += opposite_force

    total_magnitude = np.linalg.norm(total_vector)

    return total_vector, line_vectors, total_magnitude

def plot_universe(ax, window = 100): 
    # TODO separate the static vs dyanmic bodies to prevent redraws
    colors_bodies = [body.color for body in Body._instances]
    body_circles = []
    for body in Body._instances:
        circle = plt.Circle((body.x, body.y), body.radius, color=body.color)
        ax.add_patch(circle)
        body_circles.append(circle)
    
    ax.set_facecolor('black')
    
    # Set Grid and Ticks
    ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(window/10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    
    # Limits and Padding
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim = [-window,window],ylim=[-window,window]) # TODO set max environment in a more intuitive way
    plt.tight_layout(pad=0.5)

    return body_circles

# def make_on_key(ship):
#     def on_key(event):
        
#         if event.key == 'up':
#             ship.list_boosters_on['up'] = 1
    
#         elif event.key == 'down':
#             ship.list_boosters_on['down'] = 1
    
#         elif event.key == 'right':
#             ship.list_boosters_on['right'] = 1
    
#         elif event.key == 'left':
#             ship.list_boosters_on['left'] = 1
#     return on_key

def vector_field(bodies, window_size, spacing=100, max_acc=5e-3):
    """
    Compute a 2D vector field of gravitational acceleration from bodies.
    
    Parameters:
        bodies : list of Body objects
        window_size : half-width/height of the grid
        spacing : spacing between grid points
        max_acc : maximum acceleration magnitude for visualization
    
    Returns:
        X, Y : meshgrid coordinates
        U_disp, V_disp : unit direction vectors for plotting
        Mag : capped magnitude for colormap
    """
    x = np.arange(-window_size, window_size + spacing, spacing)
    y = np.arange(-window_size, window_size + spacing, spacing)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    soft = 1e-3  # small softening to prevent division by zero
    
    for body in bodies:
        if not isinstance(body, Spacecraft):
            b_x, b_y = body.x, body.y
            m = body.mass
            r_x = b_x - X
            r_y = b_y - Y
            mag_sq = np.sqrt(r_x**2 + r_y**2 + soft**2)
            
            acc_mag = GRAVITY_CONSTANT * m / (mag_sq**3)
            
            # Cap the magnitude
            acc_mag = np.minimum(acc_mag, max_acc)
            
            U += acc_mag * r_x
            V += acc_mag * r_y
    
    # Compute final magnitudes and unit vectors
    Mag = np.hypot(U, V)
    U_disp = np.nan_to_num(U / (Mag + soft))
    V_disp = np.nan_to_num(V / (Mag + soft))
    Mag = np.nan_to_num(Mag)
    
    return X, Y, U_disp, V_disp, Mag

if __name__ == "__main__":
    
    # Nate's Run Section & Scenario Setup #
    # ==================================== #
    Body._instances.clear()
    star1 = Body(name='star1', mass=12000, position=(110, 300), color='blue', radius=50, is_dynamically_updated = True)
    star2 = Body(name='star2', mass=100, position=(400, -200), velocity = (-3,.5), color='red', radius=40, is_dynamically_updated = True)
    star3 = Body(name='star3', mass=800,  position=(-350, 300), velocity = (0,2), color='yellow', radius=30, is_dynamically_updated = True)

    shipA = Spacecraft(name='spaceshipA', mass=1, position=(-600, -400), velocity = (2,0), color='white', radius = 10, thrust=1, is_dynamically_updated = True)
    target = Spacecraft(name='target', mass=1, position=(600, 400), color='purple', radius = 50, is_target=True, is_dynamically_updated = False)

    shipA.navigation_strategy = 'thrust_towards_target'

    bodies = Body._instances
    ships = Spacecraft._instances
    dt = .1  # time step

    # Example: line from bottom-left to top-right
    x_line = np.linspace(-600, 600, 40)
    y_line = np.linspace(-400, 400, 40)

    total_vec, line_vecs, mag = compute_line_gravity_cost(x_line, y_line)

    print("Total equal-and-opposite vector:", total_vec)
    print("Field imbalance cost magnitude:", mag)

    # ---- Plot using your system ----
    fig0, ax0 = plt.subplots(figsize=(6, 6), facecolor='black')
    window = 800
    plot_universe(ax0, window=window)  # reuse your helper to draw the bodies and grid
    # X,Y,U,V,M = vector_field(bodies, window, spacing = window/10)

    # Initial Plotting
    body_circles = plot_universe(ax0,window)
    # q = ax0.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', cmap='viridis', pivot='tail',zorder=-1) #TODO: Figure out scale
    plt.title("Gravitational Field Vector Field", color='white')
    # Plot the line itself
    ax0.plot(x_line, y_line, color='white', linestyle='--', linewidth=1)

    # Plot the equal-and-opposite vectors as small arrows
    for (x, y, v) in zip(x_line, y_line, line_vecs):
        ax0.arrow(x, y, v[0] * 50, v[1] * 50, head_width=10, color='cyan', alpha=0.7)

    ax0.set_title("Equal and Opposite Gravitational Field Along Line", color='white')

    # # SCENARIO SETUP #
    # # ================ #
    # scenario = 'two_body'  # Options: 'two_body', 'isaac_test1'
    # bounds = initialize_universe(scenario)
    # window=max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min)
    # bodies = Body._instances
    # ships = Spacecraft._instances

    # PLOTTING #
    # ========== #
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    # fig.canvas.mpl_connect('key_press_event', make_on_key(ships[0]))
    X,Y,U,V,M = vector_field(bodies, window, spacing = window/10)

    # Initial Plotting
    body_circles = plot_universe(ax,window)
    q = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', cmap='viridis', pivot='tail',zorder=-1) #TODO: Figure out scale
    
    path_lines = []
    for i, ship in enumerate(ships):
        path_lines.append(ax.plot([],[], color = ship.color, linewidth = 1, zorder = 0)[0])

    def update(frame):
        # Always compute physics each frame
        for body in bodies:
            if body.is_dynamically_updated:
                body.step_forward_dt(time_step = dt)
           
        for i, path in enumerate(path_lines):
            path.set_data(ships[i].path[:,0],ships[i].path[:,1])
            if ships[i].is_crashed:
                path.set_color('red')
                ships[i].is_dynamically_updated = False

        # Update vector field if any bodies are dynamic
        if any(body.is_dynamically_updated and not isinstance(body,Spacecraft) for body in bodies):   
            X,Y,U,V,M = vector_field(bodies, window, spacing = window/10)
            q.set_UVC(U, V)
            q.set_array(M.flatten())

        # Update body positions (if they move)
        for circle, body in zip(body_circles, Body._instances):
            if body.is_dynamically_updated:
                circle.center = (body.x, body.y)
        
        artists = [*path_lines, *body_circles, q]
            
        return artists
    
    ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True)

    # ani.save('simple_path_test.gif', dpi=30, writer='pillow')
    plt.show() 
    print('Fuel Spent:',ships[0].fuel_spent * dt)
    
    ## Philip's Run Section
    