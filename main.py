import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
import math
from classes import *

G = 1

def plot_universe(axes):
    
    # Structure it so we can turn on/off trails, on/off vector field, 
    # the sim will end whenever the ship is offscreen for 3 seconds
    # 
    
    # TODO separate the static vs dyanmic bodies to prevent redraws
    
    x_coord_bodies = [body.position.x for body in Body._instances]
    y_coord_bodies = [body.position.y for body in Body._instances]
    sizes_bodies = [body.mass for body in Body._instances]
    colors_bodies = [body.color for body in Body._instances]
    scatter_bodies = ax.scatter(x_coord_bodies,y_coord_bodies,s = sizes_bodies, c = colors_bodies)
    # ax.set(xlim = [min(x_coord_bodies) - 25, max(x_coord_bodies) + 25],
    #       ylim = [min(y_coord_bodies) - 25, max(y_coord_bodies) + 25],
    #        aspect = 'equal')

    return scatter_bodies

# def 2b_elliptical:
    
# def 2b_earthmoon:
    
# def 2b_follower:
    
# def 3b_figure8:
    
# def 3b_flower

# def 3b_5ptstar

# def 3b_clover

# def 3b_cross

if __name__ == "__main__":
    ## Nate's Run Section
    
    ## Isaac's Run Section
    # === Figure-eight ICs, scaled wide and stable ===
    # Masses (equal)
    m = 500.0

    # Instantiate
    Body._instances.clear()
    
    r1 = Vector2(1,3)
    v1 = Vector2(0,0)
    r2 = Vector2(50,50)
    v2 = Vector2(0,0)
    b1 = Body("b1", m, r1, v1, color="cyan")
    
    ship = Spacecraft("bro", 10, Vector2(50,0), Vector2(0,4), color="white")
    
    bodies = Body._instances
    
    # Plot Background and Grid Drawing
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.5)

    
def vector_field(bodies,spacing,bounds):
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    x = np.arange(-bounds,bounds+spacing,spacing)
    y = np.arange(-bounds,bounds+spacing,spacing)
    X,Y = np.meshgrid(x,y)
    soft = 1*10^(-9)
    for body in bodies:
        b_x = body.x
        b_y = body.y
        m = body.mass
        r_x = b_x-X
        r_y = b_y-Y
        mag_sq = np.sqrt(r_x*r_x+r_y*r_y)+soft
        acc_mag = G*m/(mag_sq)**3
        U += acc_mag*r_x
        V = acc_mag*r_y
    return X,Y,U,V
    
    # b_x = np.array([i.x for i in bodies])
    # b_y = np.array([i.y for i in bodies])
    # b_m = np.array([i.mass for i in bodies])
    # r_x = b_x[None, None, :] - X[..., None]
    # r_y = b_y[None, None, :] - X[..., None]
    
    # r2 = r_x*r_x + r_y*r_y
    # inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    # U = G * np.sum(m[None, None, :] * dx * inv_r3, axis=2)
    # V = G * np.sum(m[None, None, :] * dy * inv_r3, axis=2)
    
    
    X,Y,U,V = vector_field(bodies,100,10)
    # ax.quiver(X, Y, U, V,color='white')
    
    ## ANOTHER POSSIBLE WAY TO PLOT THE UNIVERSE
    scatter = plot_universe(ax)
    path_line, = ax.plot([], [], color='white', linewidth=1)
    trail_x, trail_y = [], []
    
    def update(frame):
        # Always compute physics each frame
        ship.step_forward_dt(time_step=1)
        b1.step_forward_dt(time_step=1)

        # Only update the plot every X frames
        if frame % 2 == 0:
            trail_x.append(ship.position.x)
            trail_y.append(ship.position.y)
            path_line.set_data(trail_x, trail_y)
            scatter.set_offsets([[b.position.x, b.position.y] for b in Body._instances])
        return (path_line, scatter)
    
    ani = animation.FuncAnimation(fig, update, frames=1000, interval=10, blit=True)
    plt.show()
    
    # ani.save('multibody_sim.gif', dpi=150, writer='pillow') 
    
    ## Philip's Run Section
    '''
    Structure
    1. Plot general universe size 
    2. Generate stable orbits
        a. Binary stars
        b. Solar systems
    4.
    How to separate planets from the ship
    How to separate stationary planets from moving planets
    '''
    