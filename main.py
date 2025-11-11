import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import math
from classes import Vector2, Body, Spacecraft, SolarSystem

def plot_universe(axes,window = 100): 
    # TODO separate the static vs dyanmic bodies to prevent redraws
    colors_bodies = [body.color for body in Body._instances]
    body_circles = []
    for body in Body._instances:
        circle = plt.Circle((body.position.x, body.position.y), body.radius, color=body.color)
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

def make_on_key(ship):
    def on_key(event):
        
        if event.key == 'up':
            ship.list_boosters_on['up'] = 1
    
        elif event.key == 'down':
            ship.list_boosters_on['down'] = 1
    
        elif event.key == 'right':
            ship.list_boosters_on['right'] = 1
    
        elif event.key == 'left':
            ship.list_boosters_on['left'] = 1
    return on_key
        

if __name__ == "__main__":
    ## Nate's Run Section
    
    ## Isaac's Run Section
    
    ## Section 2 ## Path Planning
    Body._instances.clear() # Clears any bodies left over from previous run
    star1 = Body(name= 'star1', mass = 2000, position = Vector2(0,0), velocity = Vector2(0,0),color = 'blue', radius = 50)

    shipA = Spacecraft(name ='shipA', mass = 10, position = Vector2(0,-500), velocity = Vector2(0,0), thrust = 10.0, color = 'white', radius = 10)
    target = Spacecraft(name = 'target', mass = 0, position = Vector2(-500,500), velocity = Vector2(0,0), color = 'purple', radius = 20 )

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    body_circles = plot_universe(ax,window=1000)
    
    path_x = np.arange(shipA.position.x, target.position.x, -1)
    path_y = np.arange(shipA.position.y, target.position.y, 1)
    
    ax.plot(path_x,path_y,'y--')
    
    ## Section 01 -- Starting Point ##
    # Body._instances.clear() # Clears any bodies left over from previous run
    # star1 = Body(name= 'star1', mass = 2000, position = Vector2(0,0), velocity = Vector2(0,0),color = 'blue', radius = 50)
    # # star2 = Body(name= 'star2', mass = 1500, position = Vector2(-300,700), velocity = Vector2(0,0), color = 'red', radius = 50)
    # # star3 = Body(name= 'star2', mass = 600, position = Vector2(-520,-350), velocity = Vector2(0,0), color = 'yellow', radius = 50)
    
    # spaceshipA = Spacecraft(name ='spaceshipA', mass = 10, position = Vector2(0,-500), velocity = Vector2(0,0), thrust = 10.0, color = 'white', radius = 10)
    # # target = Spacecraft(name = 'target', mass = 0, position = Vector2(-500,500), velocity = Vector2(0,0), color = 'purple', radius = 20 )
    
    # fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    # # print(spaceshipA.list_boosters_on)
    
    # ## ANOTHER POSSIBLE WAY TO PLOT THE UNIVERSE
    # body_circles = plot_universe(ax,window=1000)
    # path_line, = ax.plot([], [], color='white', linewidth=1)
    # trail_x, trail_y = [], []
    # fig.canvas.mpl_connect('key_press_event', make_on_key(spaceshipA))
    
    # def update(frame):
    #     # Always compute physics each frame
        
    #     is_crashed = spaceshipA.step_forward_dt(time_step=.5)
    
    #     if is_crashed:
    #         path_line.set_color('red')
    #         ani.event_source.stop()
            
    #     # Only update the plot every X frames
    #     if frame % 1 == 0:
    #         trail_x.append(spaceshipA.position.x)
    #         trail_y.append(spaceshipA.position.y)
            
    #         # The OG
    #         path_line.set_data(trail_x, trail_y)
            
    #         # Update body positions (if they move)
    #         for circle, body in zip(body_circles, Body._instances):
    #             circle.center = (body.position.x, body.position.y)
            
    #     return [path_line] + body_circles
    
    # ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True)
    # # plt.show()
    
    # # ani.save('gravity_sim_test_2.gif', dpi=30, writer='pillow') 
    
    
    ## Philip's Run Section
    