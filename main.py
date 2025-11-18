import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline, splev

from classes import Body, Spacecraft, GRAVITY_CONSTANT, EPSILON_GRAVITY
from utils import valid_navigation_strategies
from solar_system_config import initialize_universe

def plot_universe(ax, window=100, repulsion_factor=10.0): 
    ax.set_facecolor('black')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(window / 10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    
    colors_Bodies = [body.color for body in Bodies] 
    body_circles = []
    shadow_circles = []

    for body in Bodies:
        # main body disk
        circle = plt.Circle((body.x, body.y), body.radius, color=body.color, zorder=3)
        ax.add_patch(circle)
        body_circles.append(circle)

        # gravitational / repulsive “shadow” (only for non-spacecraft)
        if not isinstance(body, Spacecraft):
            safe_zone = body.radius * repulsion_factor
            shadow = plt.Circle(
                (body.x, body.y),
                safe_zone,
                color='red',
                alpha=0.08,
                lw=1.0,
                fill=True,
                zorder=1,
            )
            ax.add_patch(shadow)
            shadow_circles.append(shadow)

    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=[-window, window], ylim=[-window, window])
    plt.tight_layout(pad=0.5)

    return body_circles, shadow_circles

def connect_on_key_function_to_ship(ship):
    def on_key(event):
        
        if event.key == 'up':
            ship.list_boosters_on['up'] = 1
    
        elif event.key == 'down':
            ship.list_boosters_on['down'] = 1
    
        elif event.key == 'right':
            ship.list_boosters_on['right'] = 1
    
        elif event.key == 'left':
            ship.list_boosters_on['left'] = 1

        elif event.key == 't':
            ship.thrust_vec = not ship.thrust_vec
            print("Toggling Thrust Vector for Ship:", ship.name)

        elif event.key == 'v':
            ship.velocity_vec = not ship.velocity_vec
            print("Toggling Velocity Vector for Ship:", ship.name)
            
        elif event.key == 'g':
            ship.plot_vectorfield = not ship.plot_vectorfield
            print("Toggling Gravity Field On/Off")

        elif event.key == 'f':
            ship.plot_potentialfield = not ship.plot_potentialfield
            print("Toggling Potential Field Visualization")

        elif event.key == 'p':
            ship.path_visible = not ship.path_visible
            print("Toggling Path Visibility for Ships")

        elif event.key == 'l':
            ship.planet_path_visible = not ship.planet_path_visible
            print("Toggling Planet Path Visibility")

        elif event.key == 'r':
            reset_simulation()
            print("Resetting Simulation")
        
    return on_key

def vector_field(Bodies, window_size, spacing=100, max_acc=5e-4):
    """
    Compute a 2D vector field of gravitational acceleration from Bodies.
    
    Parameters:
        Bodies : list of Body objects
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
    
    for body in Bodies:
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

def body_vectors(Bodies,t_scaling=5000,v_scaling=5000):
    vel_vec = None
    thr_vec = None
    accel_vel = None
    
    for body in Bodies:
        x = body.x
        y = body.y
        dx = body.vx
        dy = body.vy
        if body.vmag != 0:
            vel_vec =  {'x':x, 'y':y,'dx':dx, 'dy':dy}
        else:
             vel_vec = {'x':x, 'y':y,'dx':0, 'dy':0}
    for body in Bodies:
        if isinstance(body,Spacecraft):
            x = body.x
            y = body.y
            dx = body.thrust[0]*t_scaling
            dy = body.thrust[1]*t_scaling
            dx_a = body.accel[0]*t_scaling
            dy_a = body.accel[1]*t_scaling
            if body.thrust_mag != 0:
                thr_vec = {'x':[x], 'y':[y],'dx':[dx], 'dy':[dy]}
                accel_vel = {'x':[x], 'y':[y],'dx_a':[dx_a], 'dy_a':[dy_a]}
            else:
                accel_vel = {'x':[x], 'y':[y],'dx_a':[0], 'dy_a':[0]}
                thr_vec = {'x':[x], 'y':[y],'dx':[0], 'dy':[0]}
    
    return vel_vec, thr_vec, accel_vel

def line(start_pt: tuple, end_pt: tuple, precision=1000, lw=3):
    x = np.linspace(start_pt[0], end_pt[0], precision)
    y = np.linspace(start_pt[1], end_pt[1], precision)
    np.stack((x,y))
    (ln,) = ax.plot(x,y,zorder=2,color='royalblue', linestyle='--',linewidth=lw, alpha = 0.5)
    return x,y,ln
    
def points_spline(x,y,precision=1000, lw=2):
    x = np.array(x)
    y = np.array(y)
    spl = CubicSpline(x, y)
    x_new = np.linspace(min(x),max(x),precision)
    plt.plot(x_new,spl(x_new),zorder=2)
    # dxdt, dydt = splev(ti, tck, der=1)

def reset_simulation():
    for body in Bodies:
        body.position[:] = body.i_p
        body.velocity[:] = body.i_v
        body.is_crashed = False
        body.is_dynamically_updated = body.i_dynamic_state
        body.path = body.i_path
    for ship in Ships:
        ship.fuel_spent = 0
        
if __name__ == "__main__": 
    
    # ============================================================================================================
    #                   S I M U L A T I O N       S E T U P
    # ============================================================================================================

    # 1. Select Scenario

    # Options '1', '2', '3', '2b_figure8', '3b_figure8', '3b_flower', '2b_figure8_chase'
    # ============================================================
    # scenario = '1'
    # scenario = '2'
    # scenario = '3'
    # scenario = '2b_figure8'
    scenario = '2b_figure8_chase'
    # scenario = '3b_figure8'
    # scenario = '3b_flower'

    scenario_bounds, Bodies, Ships = initialize_universe(scenario)
    # 2. Select Navigation Strategy

    #Options: 'stay_put', 'thrust_towards_target','line_follow', 'potential_field', 'lyapunov_pd','lyapunov_nonlinear','chase', '_'
    # =============================================================================================================
    # navigationStrategy = 'stay_put'
    # navigationStrategy = 'thrust_towards_target'
    # navigationStrategy = 'line_follow'
    navigationStrategy = 'potential_field'
    # navigationStrategy = 'lyapunov_nonlinear'
    # navigationStrategy = 'lyapunov_pd'
    # navigationStrategy = 'chase'
    # navigationStrategy = '_'

    if navigationStrategy not in valid_navigation_strategies:
        raise ValueError(f"Invalid navigation strategy: {navigationStrategy}. Must be one of {valid_navigation_strategies}")

    mainship = Ships[0]
    followPath = (-300,220) # Used for line_follow strategy, TODO: Set as optional?
    mainship.set_nav_strat(navigationStrategy,followPath)

    # 3. Select Plotting Options
    plotVectorField = True
    plotPotentialField = False
    time_step = .5

    # =============================================================================================================

    q = None
    follow_line = None
    q_v = None
    q_t = None
    q_a = None

    # PLOTTING #
    # ========== #
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black',layout='tight')
    window = max(scenario_bounds.x_max - scenario_bounds.x_min, scenario_bounds.y_max - scenario_bounds.y_min)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # Disconnect default key bindings
    fig.canvas.mpl_connect('key_press_event', connect_on_key_function_to_ship(Ships[0]))
    X,Y,U,V,M = vector_field(Bodies, window, spacing = window/10)

    # Initial Plotting
    body_circles, shadow_circles = plot_universe(ax, window, repulsion_factor=10.0)

    q = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', cmap='plasma', pivot='tail',zorder=-1)
    
    if Ships:
        if mainship.nav_strat == 'line_follow':
          x, y, follow_line = line(mainship.path_start,mainship.path_end,10000)
   
        qv, qt, qa = body_vectors([mainship])    

        # Because we have the toggle, we'll always initialize these
        q_t = ax.quiver(qt['x'],qt['y'],qt['dx'],qt['dy'], scale=1, angles='xy', scale_units='xy', color = 'orange', pivot = 'tail', zorder = 4)
        q_a = ax.quiver(qa['x'],qa['y'],qa['dx_a'],qa['dy_a'], scale=1, angles='xy', scale_units='xy', color = 'yellow', pivot = 'tail', zorder = 4)

        # Changed the scale here to be more visible
        q_v = ax.quiver(qv['x'],qv['y'],qv['dx'],qv['dy'], scale=.02, angles='xy', scale_units='xy', color = 'pink', pivot = 'tail', zorder = 4)
    
    path_lines = []
    for i, body in enumerate(Bodies):
        path_lines.append(ax.plot([],[], color = body.color, linewidth = 1.5, zorder = 0)[0])

    def update(frame):
        # Always compute physics each frame
        Body.timestep(time_step = time_step)
        for i, path in enumerate(path_lines):
            path.set_data(Bodies[i].path[:,0],Bodies[i].path[:,1])

            if isinstance(Bodies[i], Spacecraft):
                path.set_visible(mainship.path_visible)
                if Bodies[i].is_crashed:
                    path.set_color('red')
                    path.set_linestyle('--')
                else:
                    path.set_color(Bodies[i].color)
                path.set_linestyle('-')
            else :
                path.set_visible(mainship.planet_path_visible)
                if Bodies[i].is_dynamically_updated == False:
                    path.set_data([],[])
                
        
        # Update vector field if any Bodies are dynamic
        q.set_visible(mainship.plot_vectorfield)
        if mainship.plot_vectorfield:
            if any(body.is_dynamically_updated and not isinstance(body,Spacecraft) for body in Bodies):   
                X,Y,U,V,M = vector_field(Bodies, window, spacing = window/10)
                q.set_UVC(U, V)
                q.set_array(M.flatten())
                
        # Update body positions (if they move)
        for circle, body in zip(body_circles, Bodies):
            if body.is_dynamically_updated:
                circle.center = (body.x, body.y)
        
        # Update shadow rings
        
        for shadow, body in zip(shadow_circles, [b for b in Bodies if not isinstance(b, Spacecraft)]):
            if body.is_dynamically_updated:
                shadow.center = (body.x, body.y)
            shadow.set_visible(mainship.plot_potentialfield)

        if Ships:
            qv,qt,qa = body_vectors([mainship])
            q_v.set_visible(mainship.velocity_vec)
            q_v.set_UVC(qv['dx'], qv['dy'])
            q_v.set_offsets(np.array([[qv['x'], qv['y']]]))

            q_t.set_visible(mainship.thrust_vec and mainship.thrust_mag != 0)
            q_t.set_UVC(qt['dx'], qt['dy'])
            q_t.set_offsets(np.array([[qt['x'], qt['y']]]))
            q_a.set_UVC(qa['dx_a'], qa['dy_a'])
            q_a.set_offsets(np.array([[qa['x'], qa['y']]]))
        
        pot_artists = [*path_lines, *body_circles, *shadow_circles, q, follow_line, q_t, q_v, q_a]  
        
        artists = [artist for artist in pot_artists if artist is not None]
            
        return artists
    
    ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True, repeat=True)
    # update(0) # Debug requires this to not run on GUI loop

    # ani.save('flowertest.gif', dpi=100, writer='pillow')
    plt.show() 
    