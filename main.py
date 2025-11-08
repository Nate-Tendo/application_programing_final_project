import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
import math
from classes import *

def plot_universe(axes):
    
    # TODO separate the static vs dyanmic bodies to prevent redraws
    
    x_coord_bodies = [body.position.x for body in Body._instances]
    y_coord_bodies = [body.position.y for body in Body._instances]
    sizes_bodies = [body.mass for body in Body._instances]
    colors_bodies = [body.color for body in Body._instances]
    scatter_bodies = ax.scatter(x_coord_bodies,y_coord_bodies,s = sizes_bodies, c = colors_bodies)
    ax.set(xlim = [min(x_coord_bodies) - 25, max(x_coord_bodies) + 25],
           ylim = [min(y_coord_bodies) - 25, max(y_coord_bodies) + 25],
           aspect = 'equal')

    return scatter_bodies

if __name__ == "__main__":
    ## Nate's Run Section
    
    ## Isaac's Run Section
    Body._instances.clear() # Clears any bodies left over from previous run
    star1 = Body(name= 'star1', mass = 500, position = Vector2(0,0), velocity = Vector2(0,0),color = 'blue')
    star2 = Body(name= 'star2', mass = 500, position = Vector2(100,100), velocity = Vector2(0,0), color = 'red')
    
    spaceshipA = Spacecraft(name ='spaceshipA', mass = 10, position = Vector2(100,0), velocity = Vector2(-1,0),color = 'white')
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)
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
    
    ## ONE POSSIBLE WAY TO PLOT HTE UNIVERSE
    # plot_universe(ax)
    # for i in range(1000): spaceshipA.step_forward_dt(time_step = .1)
    # xs = [p.x for p in spaceshipA.path]
    # ys = [p.y for p in spaceshipA.path]
    # ax.plot(xs, ys,'b-o',markersize = 1)
    
    
    ## ANOTHER POSSIBLE WAY TO PLOT THE UNIVERSE
    scatter = plot_universe(ax)
    path_line, = ax.plot([], [], color='white', linewidth=1)
    trail_x, trail_y = [], []
    
    def update(frame):
        # Always compute physics each frame
        spaceshipA.step_forward_dt(time_step=1)
    
        # Only update the plot every X frames
        if frame % 1 == 0:
            trail_x.append(spaceshipA.position.x)
            trail_y.append(spaceshipA.position.y)
            path_line.set_data(trail_x, trail_y)
            scatter.set_offsets([[b.position.x, b.position.y] for b in Body._instances])
        return (path_line, scatter)
    
    ani = animation.FuncAnimation(fig, update, frames=2000, interval=5, blit=True)
    # plt.show()
    
    # ani.save('gravity_sim_test_1.gif', dpi=80, writer='pillow') 
    
    ## Philip's Run Section
    