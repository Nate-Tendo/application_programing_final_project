import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import math
from physics_classes import Bounds, CelestialBody, VehicleBody, Universe, load_universe_from_config
from visualization import Renderer 

def main():
    universe = load_universe_from_config("config_files/config1.json")

    print("=== Universe Configuration ===")
    print(f"Bounds: {universe.bounds}")
    print("\nBodies:")
    for body in universe.all_bodies:
        print(f"  {body.name}: pos={body.position}, vel={body.velocity}, mass={body.mass}, radius={body.radius}")

    print("\nShips:")
    for ship in universe.ships:
        print(f"  {ship.name}: pos={ship.position}, vel={ship.velocity}, max_thrust={ship.max_thrust}")
    
    return universe

if __name__ == "__main__":
    universe = main()

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    colors_bodies = [body.color for body in universe.all_bodies]
    body_circles = []
    for body in universe.all_bodies:
        circle = plt.Circle((body.position[0], body.position[1]), body.radius, color=body.color)
        ax.add_patch(circle)
        body_circles.append(circle)

    ax.set_facecolor('black')

    window = abs(universe.bounds.x_max - universe.bounds.x_min)
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

    path_line, = ax.plot([], [], color='white', linewidth=1)
    trail_x, trail_y = [], []

    spaceshipA = universe.ships[0]

    def update(frame):
        # Always compute physics each frame
        
        is_crashed = spaceshipA.step_forward_dt(time_step=.5)

        if is_crashed:
            path_line.set_color('red')
            ani.event_source.stop()
            
        # Only update the plot every X frames
        if frame % 1 == 0:
            trail_x.append(spaceshipA.position.x)
            trail_y.append(spaceshipA.position.y)
            
            # The OG
            path_line.set_data(trail_x, trail_y)
            
            # Update body positions (if they move)
            for circle, body in zip(body_circles, universe.all_bodies):
                circle.center = (body.position[0], body.position[1])
            
        return [path_line] + body_circles

    ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True)
