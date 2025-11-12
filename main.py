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

    Body._instances.clear()
    
    star1 = Body(name= 'star1', mass = 500, position = Vector2(-100,0), velocity = Vector2(0,1.11803398875),color = 'blue')
    star2 = Body(name= 'star2', mass = 500, position = Vector2(100,0), velocity = Vector2(0,-1.11803398875), color = 'red')
    
    spaceshipA = Spacecraft(name ='spaceshipA', mass = 10, position = Vector2(0,0), velocity = Vector2(1,0),color = 'white',)
    bodies = Body._instances
    
    # Plot Background and Grid Drawing
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    # ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.5)
    
    def vector_field(bodies):
        bounds = 200
        spacing = 25
        x = np.arange(-bounds,bounds+spacing,spacing)
        y = np.arange(-bounds,bounds+spacing,spacing)
        X,Y = np.meshgrid(x,y)
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        soft = 1*10**(-3)
        for body in bodies:
            if not isinstance(body,Spacecraft):
                b_x = body.position.x
                b_y = body.position.y
                m = body.mass
                r_x = b_x-X
                r_y = b_y-Y
                mag_sq = np.sqrt(r_x*r_x+r_y*r_y+soft*soft)
                acc_mag = G*m/(mag_sq)**3
                U += acc_mag*r_x
                V += acc_mag*r_y
                U = np.nan_to_num(U)
                V = np.nan_to_num(V)
                Mag = np.hypot(U, V)
                U_disp = np.nan_to_num(U/(Mag+soft)) 
                V_disp = np.nan_to_num(V/(Mag+soft))
                Mag = np.nan_to_num(Mag)
        return X,Y,U_disp,V_disp,Mag
      
    X,Y,U,V,M = vector_field(bodies)
    
    
    # Initial Plotting
    q = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', scale=0.08, cmap='plasma', pivot='tail',zorder=-1)
    scatter = plot_universe(ax)
    path_line, = ax.plot([], [], color='white', linewidth=1,zorder=0)
    trail_x, trail_y = [], []
    
    
# ------------------------------------------------------------------------------    
    # Update Plotting
    def update(frame):
        # Always compute physics each frame
        spaceshipA.step_forward_dt(time_step=1)
        star2.step_forward_dt(time_step=1)
        star1.step_forward_dt(time_step=1)
  
        X,Y,U,V,M = vector_field(bodies)
        
        # Only update the plot every X frames
        if frame % 1 == 0:
            trail_x.append(spaceshipA.position.x)
            trail_y.append(spaceshipA.position.y)
            path_line.set_data(trail_x, trail_y)
            scatter.set_offsets([[b.position.x, b.position.y] for b in Body._instances])
            q.set_UVC(U, V)
            q.set_array(M.flatten())
        return (path_line, scatter, q)
    
    ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True)
    plt.show()
    
    # ani.save('multibody_sim.gif', dpi=50, writer='pillow', fps=60) 
    
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
    