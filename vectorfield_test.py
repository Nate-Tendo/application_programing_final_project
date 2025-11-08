import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import integrate

# Constants
G = 6.6743*10**(-11) # (m^3⋅kg^−1⋅s^−2) - Gravitational constant

plt.close('all') 

# def gravity_vector(p1,p2,m):
    
    
class vector():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        return (f'({self.x},{self.y})')
    def __add__(self,other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        return vector(new_x,new_y)
    def __sub__(self,other):
        new_x = self.x - other.x
        new_y = self.y - other.y
        return vector(new_x,new_y)
    def __rsub__(self,other):
        new_x = -self.x + other.x
        new_y = -self.y + other.y
        return vector(new_x,new_y)
    def __mul__(self,scalar):
        return vector(self.x*scalar,self.y*scalar)
    def __rmul__(self,scalar):
        return vector(self.x*scalar,self.y*scalar)
    def mag(self):
        return (np.sqrt(self.x**2+self.y**2))
    def unit_vec(self):
        mag = self.mag()
        if mag == 0:
            return vector(0,0)
        return vector(self.x/mag,self.y/mag)
    def angle(self):
        return np.arctan2(self.y,self.x)        
        
class body():
    _instances = []
    def __init__(self,x_i,y_i,vx_i,vy_i,m):
        self.m = m
        self.x = x_i
        self.y = y_i
        self.p = vector(self.x,self.y)
        body._instances.append(self)
    def __str__(self):
        return (f'Position: ({self.x},{self.y})\nMass: {self.m}')
    def plot(self):
        plt.plot(self.x,self.y,'.',markersize=self.m)
    def pos(self):
        return vector(self.x,self.y)
    def plot_vectorfield(self,spacing,boundaries):
        x = np.arange(-boundaries, boundaries+spacing, spacing)
        y = np.arange(-boundaries, boundaries+spacing, spacing)
        X, Y = np.meshgrid(x, y)
        
        # -G*self.m/(r.magnitude())**2*r.unit_vec()
        U = -G*self.m/(self.p.mag())**2*self.p.unit_vec()
        V = -G*self.m/(self.p.mag())**2*self.p.unit_vec()
        return(X,Y,U,V)

# make data
# x = np.arange(-50, 65, 5)
# y = np.arange(-50, 65, 5, )
# X, Y = np.meshgrid(x, y)
# U = -X
# V = Y

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

# ax.quiver(X, Y, U, V,color='white')

Moon = body(0,0,0,0,200)
Moon.plot()

plt.show()