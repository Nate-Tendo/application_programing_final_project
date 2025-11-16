import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
import matplotlib.collections # import PatchCollection
import math
from classes import Body, Spacecraft, GRAVITY_CONSTANT, EPSILON_GRAVITY
from solar_system_config import initialize_universe
from scipy.interpolate import CubicSpline, splev

def compute_line_gravity_cost(x_coords: np.ndarray, y_coords: np.ndarray):
    """
    Compute the gravitational 'cost' along a line defined by x,y coordinates.

    Parameters
    ----------
    x_coords, y_coords : array-like
        Arrays (or lists) of x and y coordinates along the line. Must have the same length.

    Returns
    -------
    total_vector : np.array
        Sum of all equal-and-opposite gravitational vectors (represents total imbalance).
    line_vectors : list of np.array
        The opposite-force vectors at each line point.
    total_magnitude : float
        The magnitude of the summed total_vector (scalar cost).
    """

    assert len(x_coords) == len(y_coords), "x_coords and y_coords must be the same length."

    total_vector = np.array((0.0, 0.0),dtype=float)
    line_vectors = []

    for x, y in zip(x_coords, y_coords):
        point = np.array((x, y),dtype=float)
        total_gravity = np.array((0.0, 0.0),dtype=float)
        for body in Body._instances:
            direction = body.position - point
            distance = np.linalg.norm(direction)
            if distance < EPSILON_GRAVITY:
                continue
            g_force_mag = GRAVITY_CONSTANT * body.mass / (distance**2 + EPSILON_GRAVITY**2)
            total_gravity += (direction / distance) * g_force_mag
        # Equal and opposite (to make net zero)
        opposite_force = total_gravity * -1
        line_vectors.append(opposite_force)
        total_vector += opposite_force

    total_magnitude = np.linalg.norm(total_vector)

    return total_vector, line_vectors, total_magnitude

def plot_universe(ax, window = 100): 
    ax.set_facecolor('black')
    
    # Set Grid and Ticks
    # ax.grid(True, which='both', color='white', alpha=0.25, linewidth=0.8, zorder=-3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(window/10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    
    # TODO separate the static vs dyanmic bodies to prevent redraws
    colors_bodies = [body.color for body in Body._instances]
    body_circles = []
    for body in Body._instances:
        circle = plt.Circle((body.x, body.y), body.radius, color=body.color)
        ax.add_patch(circle)
        body_circles.append(circle)
    
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

def vector_field(bodies, window_size, spacing=100, max_acc=5e-4):
    """
    Compute a 2D vector field of gravitational acceleration from bodies.
    
    Parameters:
        bodies : list of Body objects
        window_size : half-width/height of the grid
        spacing : spacing between grid points
        max_acc : maximum acceleration magnitude for visualization
    
    Returns:
        X, Y : meshgrid coordinates
        U_disp, V_disp : unit direction vectors for plotting
        Mag : capped magnitude for colormap
    """
    x = np.arange(-window_size, window_size + spacing, spacing)
    y = np.arange(-window_size, window_size + spacing, spacing)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    soft = 1e-3  # small softening to prevent division by zero
    
    for body in bodies:
        if not isinstance(body, Spacecraft):
            b_x, b_y = body.x, body.y
            m = body.mass
            r_x = b_x - X
            r_y = b_y - Y
            mag_sq = np.sqrt(r_x**2 + r_y**2 + soft**2)
            
            acc_mag = GRAVITY_CONSTANT * m / (mag_sq**3)
            
            # Cap the magnitude
            acc_mag = np.minimum(acc_mag, max_acc)
            
            U += acc_mag * r_x
            V += acc_mag * r_y
    
    # Compute final magnitudes and unit vectors
    Mag = np.hypot(U, V)
    U_disp = np.nan_to_num(U / (Mag + soft))
    V_disp = np.nan_to_num(V / (Mag + soft))
    Mag = np.nan_to_num(Mag)
    
    return X, Y, U_disp, V_disp, Mag

def body_vectors(bodies,t_scaling=5000,v_scaling=5000):
    vel_vec = None
    thr_vec = None
    
    for body in bodies:
        x = body.x
        y = body.y
        dx = body.vx
        dy = body.vy
        if body.vmag != 0:
            vel_vec =  {'x':x, 'y':y,'dx':dx, 'dy':dy}
        else:
             vel_vec = {'x':x, 'y':y,'dx':0, 'dy':0}
    for body in bodies:
        if isinstance(body,Spacecraft):
            # print('I got it')
            x = body.x
            y = body.y
            dx = body.thrust[0]*t_scaling
            dy = body.thrust[1]*t_scaling
            # print(np.array([dx,dy]))
            if body.thrust_mag != 0:
                thr_vec = {'x':[x], 'y':[y],'dx':[dx], 'dy':[dy]}
            else:
                thr_vec = {'x':[x], 'y':[y],'dx':[0], 'dy':[0]}
    return vel_vec, thr_vec

r = np.arange(-2,4,0.01)
def f_1(r):
    x = 50*(r**3 - 5*r**2 +3*r + 11)+200
    y = 50*(r**2 - 2*r + 3)
    return x,y


def parametric_func(f,r,lw=3):
    x,y = f(r)
    ax.plot(x,y,zorder=2,linewidth=lw,linestyle='--')

def line(start_pt: tuple, end_pt: tuple, precision=1000, lw=3):
    x = np.linspace(start_pt[0], end_pt[0], precision)
    y = np.linspace(start_pt[1], end_pt[1], precision)
    np.stack((x,y))
    (ln,) = ax.plot(x,y,zorder=2,color='powderblue', linestyle='--',linewidth=lw)
    return x,y,ln
    
def points_spline(x,y,precision=1000, lw=2):
    x = np.array(x)
    y = np.array(y)
    spl = CubicSpline(x, y)
    x_new = np.linspace(min(x),max(x),precision)
    plt.plot(x_new,spl(x_new),zorder=2)
    # dxdt, dydt = splev(ti, tck, der=1)

    q = None
    follow_line = None
        
if __name__ == "__main__":
    

        
    
    scenario = '2b_figure8'
    bounds = initialize_universe(scenario)
    window=max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min)
    bodies = Body._instances
    plotVectorField = True
    ships = Spacecraft._instances
    dt = 10
    if ships:
        mainship = ships[0]
        mainship.set_nav_strat('line_follow',(200,-215))
        # mainship.set_nav_strat('stay_put')
        '''
        # mainship.set_nav_strat('path_follow',
                                 [(),(),(),(),()]
        '''
   
    
   
    # PLOTTING #
    # ========== #
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    # fig.canvas.mpl_connect('key_press_event', make_on_key(ships[0]))
    X,Y,U,V,M = vector_field(bodies, window, spacing = window/10)

    # Initial Plotting
    body_circles = plot_universe(ax,window)
    if plotVectorField == True:
        q = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', cmap='plasma', pivot='tail',zorder=-1)
    
    if mainship.nav_strat == 'line_follow':
      x, y, follow_line = line(mainship.path_start,mainship.path_end,10000)

    # points_spline(, y)

    # parametric_func(f_1,r)
    
    qv, qt = body_vectors([mainship])    
    if mainship.thrust_vec == True:
        q_t = ax.quiver(qt['x'],qt['y'],qt['dx'],qt['dy'], scale=1, angles='xy', scale_units='xy', color = 'orange', pivot = 'tail', zorder = 4)
    if mainship.thrust_vec == True:
        q_v = ax.quiver(qv['x'],qv['y'],qv['dx'],qv['dy'], scale=1, angles='xy', scale_units='xy', color = 'skyblue', pivot = 'tail', zorder = 4)
    
    path_lines = []
    for i, ship in enumerate(ships):
        path_lines.append(ax.plot([],[], color = 'white', linewidth = 3, zorder = 0)[0])

    def update(frame):
        # Always compute physics each frame
        Body.timestep(dt)
               
        for i, path in enumerate(path_lines):
            path.set_data(ships[i].path[:,0],ships[i].path[:,1])
            # if ships[i].is_crashed:
            #     path.set_color('red')
            #     for ship in ships:
            #         ship.is_dynamically_updated = False
        
        if frame % 1 == 0:
            # Update vector field if any bodies are dynamic
            if plotVectorField == True:
                if any(body.is_dynamically_updated and not isinstance(body,Spacecraft) for body in bodies):   
                    X,Y,U,V,M = vector_field(bodies, window, spacing = window/10)
                    q.set_UVC(U, V)
                    q.set_array(M.flatten())
                    
            # Update body positions (if they move)
            for circle, body in zip(body_circles, Body._instances):
                if body.is_dynamically_updated:
                    circle.center = (body.x, body.y)
            
            qv,qt = body_vectors([ships[0]])
            if mainship.velocity_vec == True:
                q_v.set_UVC(qv['dx'], qv['dy'])
                q_v.set_offsets(np.array([[qv['x'], qv['y']]]))
            if mainship.thrust_vec == True:
                q_t.set_UVC(qt['dx'], qt['dy'])
                q_t.set_offsets(np.array([[qt['x'], qt['y']]]))
            
            pot_artists = [*path_lines, *body_circles, q, follow_line,q_t]  
            
            artists = [artist for artist in pot_artists if artist is not None]
            
            if frame == 200:
                print('Fuel Spent:',ships[0].fuel_spent)
            
        return artists
    
    ani = animation.FuncAnimation(fig, update, frames=1000, interval=1, blit=True, repeat=True)

    # ani.save('simple_path_test.gif', dpi=100, writer='pillow')
    plt.show() 
    # print('Fuel Spent:',ships[0].fuel_spent)
    # print('Fuel Spent:',ships[0].fuel_spent)
    
    ## Philip's Run Section
    