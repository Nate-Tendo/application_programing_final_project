# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

class Renderer:
    def __init__(self, universe):
        self.universe = universe
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        plt.ion()

    def draw(self, predicted_path=None):
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.set_xlim(self.universe.bounds[0], self.universe.bounds[1])
        self.ax.set_ylim(self.universe.bounds[2], self.universe.bounds[3])

        # Draw celestial bodies
        for b in self.universe.bodies:
            self.ax.scatter(*b.position, color=b.color, s=50)

        # Draw ship(s)
        for s in self.universe.ships:
            trail = np.array(s.trail)
            self.ax.plot(trail[:,0], trail[:,1], color=s.color, alpha=0.5)

            size = 2e7
            triangle = np.array([[0, size*0.5], [-size*0.3, -size*0.5], [size*0.3, -size*0.5]])
            rot = np.array([[np.cos(s.orientation), -np.sin(s.orientation)],
                            [np.sin(s.orientation), np.cos(s.orientation)]])
            triangle = triangle @ rot.T + s.position
            self.ax.add_patch(Polygon(triangle, closed=True, color=s.color))

            # Thrust flame if active
            if s.thrust_on:
                flame = np.array([[0, -size*0.8], [-size*0.1, -size*0.5], [size*0.1, -size*0.5]])
                flame = flame @ rot.T + s.position
                self.ax.add_patch(Polygon(flame, closed=True, color='orange'))

        # Predicted trajectory
        if predicted_path is not None:
            self.ax.plot(predicted_path[:,0], predicted_path[:,1], '--', color='cyan')

        plt.pause(0.001)
