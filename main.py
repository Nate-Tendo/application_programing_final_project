import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
import math
import time
import numpy as np
from classes import Universe, Spaceship, PhysicsEngine, TrajectoryPredictor
from visualization import Renderer
def plot_universe(axes):
    
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

def vector_field(bodies, window_size, spacing=100, max_acc=5e-4):
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
    ## Nate's Run Section
        # Create environment
    universe = Universe(num_bodies=4)
    ship = Spaceship("Explorer", position=[-8e8, -8e8], velocity=[0, 0], color='white')
    universe.add_ship(ship)

    goal = np.array([8e8, 8e8])
    predictor = TrajectoryPredictor(universe)
    predicted_path = predictor.predict(ship, goal)

    renderer = Renderer(universe)

    dt = 1000
    thrust_mag = 5e-3

    for _ in range(20000):
        to_goal = goal - ship.position
        dist = np.linalg.norm(to_goal)
        if dist < 2e7:
            break

        # Steering and thrust
        dir_to_goal = to_goal / dist
        thrust = thrust_mag * dir_to_goal
        ship.thrust_on = True
        ship.orientation = np.arctan2(dir_to_goal[1], dir_to_goal[0])

        # Update physics
        PhysicsEngine.integrate_rk4(ship, universe.bodies, dt, thrust)
        ship.update_trail()

        renderer.draw(predicted_path)
        time.sleep(0.01)

    ship.thrust_on = False
    renderer.draw(predicted_path)
    print("Arrived near goal.")
    
    # ## Isaac's Run Section
    # Body._instances.clear() # Clears any bodies left over from previous run
    # star1 = Body(name= 'star1', mass = 500, position = Vector2(0,0), velocity = Vector2(0,0),color = 'blue')
    # star2 = Body(name= 'star2', mass = 500, position = Vector2(100,100), velocity = Vector2(0,0), color = 'red')
    
    # spaceshipA = Spacecraft(name ='spaceshipA', mass = 10, position = Vector2(100,0), velocity = Vector2(-1,0),color = 'white')
    
    # fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    # ax.set_facecolor('black')
    # # ax.set_xlim(-50, 50)
    # # ax.set_ylim(-50, 50)
    # ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_frame_on(False)
    # ax.tick_params(tick1On=False)
    # locator = MultipleLocator(10)
    # ax.xaxis.set_major_locator(locator)
    # ax.yaxis.set_major_locator(locator)
    # ax.set_aspect('equal', adjustable='box')

    # plt.tight_layout(pad=0.5)
    
    # ## ONE POSSIBLE WAY TO PLOT HTE UNIVERSE
    # # plot_universe(ax)
    # # for i in range(1000): spaceshipA.step_forward_dt(time_step = .1)
    # # xs = [p.x for p in spaceshipA.path]
    # # ys = [p.y for p in spaceshipA.path]
    # # ax.plot(xs, ys,'b-o',markersize = 1)
    
    
    # ## ANOTHER POSSIBLE WAY TO PLOT THE UNIVERSE
    # scatter = plot_universe(ax)
    # path_line, = ax.plot([], [], color='white', linewidth=1)
    # trail_x, trail_y = [], []
    
    # def update(frame):
    #     # Always compute physics each frame
    #     spaceshipA.step_forward_dt(time_step=1)
    
    #     # Only update the plot every X frames
    #     if frame % 1 == 0:
    #         trail_x.append(spaceshipA.position.x)
    #         trail_y.append(spaceshipA.position.y)
    #         path_line.set_data(trail_x, trail_y)
    #         scatter.set_offsets([[b.position.x, b.position.y] for b in Body._instances])
    #     return (path_line, scatter)
    
    # ani = animation.FuncAnimation(fig, update, frames=2000, interval=5, blit=True)
    # # plt.show()
    
    # # ani.save('gravity_sim_test_1.gif', dpi=80, writer='pillow') 
    
    # ## Philip's Run Section
