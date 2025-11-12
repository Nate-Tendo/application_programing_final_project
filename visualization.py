# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

class Renderer:
    def __init__(self, universe):
        self.universe = universe
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(universe.bounds.x_min, universe.bounds.x_max)
        self.ax.set_ylim(universe.bounds.y_min, universe.bounds.y_max)

        # Grid setup
        locator = MultipleLocator((universe.bounds.x_max - universe.bounds.x_min) / 10)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.yaxis.set_major_locator(locator)
        self.ax.grid(True, which="both", color="white", alpha=0.25, linewidth=0.8)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.ax.set_frame_on(False)

        # Initialize body patches
        self.body_patches = []
        for body in universe.all_bodies:
            circle = Circle((body.position['x'], body.position['y']), body.radius, color=body.color)
            self.ax.add_patch(circle)
            self.body_patches.append((circle, body))

    def draw(self, predicted_path=None):
        # Update positions of all bodies
        for circle, body in self.body_patches:
            circle.center = (body.position['x'], body.position['y'])

