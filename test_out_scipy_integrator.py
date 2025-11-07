from scipy.integrate import solve_ivp
import numpy as np
import collections
import matplotlib.pyplot as plt
import math

GRAVITY_CONSTANT = 10
planet_mass = 10
Planet = collections.namedtuple('Planet', ('x','y','m'))
Spaceship = collections.namedtuple('Spaceship',('x','y','vx','vy','m'))
planet1 = Planet(0,0,planet_mass)
ship1 = Spaceship(x=500,y=500,vx=-.1,vy=.1,m=1)


# def get_gravity_field(x,y):
    
#     return fx, fy

# def draw_gravity_field():
#     grix_x_y --> get_gravity(allx,allys)
#     draws().


def state_derivatives(t,y):
    x,y,vx,vy = y

    angle = math.atan2(planet1.y - y, planet1.x - x)
    distance = math.dist((planet1.x,planet1.y),(x,y))
    
    dxdt = vx
    dydt = vy
    dvxdt = math.cos(angle)* GRAVITY_CONSTANT * planet1.m/distance ** 2
    dvydt = math.sin(angle)* GRAVITY_CONSTANT * planet1.m/distance ** 2
    return [dxdt, dydt, dvxdt, dvydt]
    
fig, ax = plt.subplots()

ax.scatter(planet1.x,planet1.y,s=100,c='g')
ax.scatter(ship1.x,ship1.y,s=10,c='b')

tspan = (0,30000)
t_range = np.arange(tspan[0],tspan[1],.1)

ax.set(xlim=[-1000,1000],ylim=[-1000,1000])

path = solve_ivp(fun=state_derivatives,t_span=tspan,t_eval = t_range, y0=[ship1.x,ship1.y,ship1.vx,ship1.vy],rtol=1e-8, atol=1e-8)

ax.plot(path.y[0],path.y[1],'r-', linewidth = 1)

