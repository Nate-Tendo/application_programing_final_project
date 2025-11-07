import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import integrate

plt.close('all') 

def gravity_vector(p1,p2,m):
    pass
    
class vector():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self,x,y):
        return (self.x,self.y)
    def __add__(self,other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        return vector(new_x,new_y)
    def __sub__(self,other):
        new_x = self.x - other.x
        new_y = self.y - other.y
        return vector(new_x,new_y)
    def magnitude(self):
        return (np.sqrt(self.x**2+self.y**2))

class body():
    _instances = []
    def __init__(self,x_i,y_i,vx_i,vy_i,m):
        self.m = m
        self.x = x_i
        self.y = y_i
        body._instances.append(self)
    def plot(self):
        plt.plot(self.x,self.y,'.',markersize=self.m)
    def vectorfield(self):
        pass

# Constants
G = 6.6743*10**(-11) # (m^3⋅kg^−1⋅s^−2) - Gravitational constant

# make data
x = np.arange(-45, 50, 5)
y = np.arange(-45, 50, 5, )
X, Y = np.meshgrid(x, y)
U = X
V = Y

# plot
fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)


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

ax.quiver(X, Y, U, V,color='white')

Moon = body(0,0,0,0,200)
Moon.plot()

plt.show()