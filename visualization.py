import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline, splev

from classes import Body, Spacecraft, GRAVITY_CONSTANT, EPSILON_GRAVITY
from utils import valid_navigation_strategies
        

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
            else:
                thr_vec = {'x':[x], 'y':[y],'dx':[0], 'dy':[0]}
    
    return vel_vec, thr_vec

def line(ax,start_pt: tuple, end_pt: tuple, precision=1000, lw=3):
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

def plot_universe(ax, Bodies, window=100, repulsion_factor=10.0): 
    ax.set_facecolor('black')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    locator = MultipleLocator(window / 10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    
    body_circles = []
    shadow_circles = []
    path_lines = []

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
                color='pink',
                alpha=0.18,
                lw=1.0,
                fill=True,
                zorder=1,
            )
            ax.add_patch(shadow)
            shadow_circles.append(shadow)

        path_lines.append(ax.plot([],[], color = body.color, linewidth = 1.5, zorder = 0)[0])

    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=[-window, window], ylim=[-window, window])
    plt.tight_layout(pad=0.5)

    return body_circles, shadow_circles, path_lines

def connect_on_key_function_to_ship(ship,settings,Bodies,Ships):
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
            settings.thrust_vector_on = not settings.thrust_vector_on
            print("Toggling Thrust Vector")

        elif event.key == 'v':
            settings.vel_vector_on = not settings.vel_vector_on
            print("Toggling Velocity Vector")
            
        elif event.key == 'g':
            settings.vector_field_on = not settings.vector_field_on
            print("Toggling Gravity Field On/Off")

        elif event.key == 'f':
            settings.potential_field_on = not settings.potential_field_on
            print("Toggling Potential Field Visualization")

        elif event.key == 'p':
            settings.ship_path_on = not settings.ship_path_on
            print("Toggling Path Visibility for Ships")

        elif event.key == 'l':
            settings.planet_path_on = not settings.planet_path_on
            print("Toggling Planet Path Visibility")

        elif event.key == 'r':
            reset_simulation(Bodies,Ships)
            print("Resetting Simulation")

        elif event.key == 'h':
            settings.help_on = not settings.help_on
            print("Toggling Help Menu On/Off")

        elif event.key == 's':
            settings.rel_stats_on = not settings.rel_stats_on
            print("Toggling Relative Stats On/Off")
        
    return on_key

def reset_simulation(Bodies,Ships):
    global current_time
    current_time = 0.0 # Reset time counter
    for body in Bodies:
        body.position[:] = body.i_p
        body.velocity[:] = body.i_v
        body.is_crashed = False
        body.is_dynamically_updated = body.i_dynamic_state
        body.path = body.i_path
    for ship in Ships:
        ship.fuel_spent = 0

class PlotSettings:
    
    def __init__(self,vector_field = True, ship_path = True, potential_field = False, vel_vector = False, thrust_vector = False, planet_path = False, help_on = True, rel_stats_on = True):
        self.vector_field_on = vector_field
        self.ship_path_on = ship_path
        self.potential_field_on = potential_field
        self.vel_vector_on = vel_vector
        self.thrust_vector_on = thrust_vector
        self.planet_path_on = planet_path
        self.help_on = help_on
        self.rel_stats_on = rel_stats_on

def plot_universe_animation(Bodies, Ships, Scenario_Bounds, Time_Step, navigationStrategy = '_', follow_path = (0,0),scenario_name = "Unnamed Scenario"):

    # Create and Initialize Variables and Settings #
    # ======================================== #


    settings = PlotSettings(vector_field = True, ship_path = True, help_on = True, rel_stats_on = False,
                            potential_field = False, vel_vector = False, thrust_vector = False, planet_path = False)

    q = None
    follow_line = None
    q_v = None
    q_t = None

    mainship = Ships[0]
    if navigationStrategy not in valid_navigation_strategies: raise ValueError(f"Invalid navigation strategy: {navigationStrategy}. Must be one of {valid_navigation_strategies}")
    mainship.set_nav_strat(navigationStrategy, follow_path)

    time_step = Time_Step
    
    # ========== #
    #  PLOTTING  #
    # ========== #
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='black',layout='tight')
    window = max(Scenario_Bounds.x_max - Scenario_Bounds.x_min, Scenario_Bounds.y_max - Scenario_Bounds.y_min)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # Disconnect default key bindings
    fig.canvas.mpl_connect('key_press_event', connect_on_key_function_to_ship(mainship,settings,Bodies,Ships))
    # ================= #
    # Initial Plotting
    body_circles, shadow_circles, path_lines = plot_universe(ax, Bodies, window, repulsion_factor=10.0)

    # Labels #

    time_label = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=9,fontfamily='monospace',color='white',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

    toggle_label = ax.text(0, 0,
        'Control Toggles:\n'
        '\n h — Help Menu'
        '\n t — Thrust Vector'
        '\n v — Velocity Vector'
        '\n g — Gravity Field'
        '\n f — Potential Field'
        '\n p — Ship Paths'
        '\n l — Planet Paths'
        '\n r — Reset Simulation'
        '\n s — Relative Stats',
        transform=ax.transAxes, color='white', fontsize=9, fontfamily='monospace', va='bottom',
        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.4'))
    
    relative_position_label = ax.text(
        .99, .99,'',transform=ax.transAxes, ha='right', va='top',
        fontsize=9,fontfamily='monospace',color='white',
        bbox=dict(facecolor='black', alpha=0.6,boxstyle='round,pad=0.5'))
    
    title_label = ax.text(
        1, 0,f"{scenario_name} — Navigation Strategy: {navigationStrategy}",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9,fontfamily='monospace',color='white',
        bbox=dict(facecolor='black', alpha=0.6,boxstyle='round,pad=0.5'))
    

    # Vector Plotting #
    X,Y,U,V,M = vector_field(Bodies, window, spacing = window/10)
    q = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', cmap='plasma', pivot='tail',zorder=-1)

    if Ships:
        if mainship.nav_strat == 'line_follow':
          x, y, follow_line = line(ax,mainship.path_start,mainship.path_end,10000)
   
        qv, qt = body_vectors([mainship])    

        if navigationStrategy == 'manual_boosters':
            thrust_scale = 100
        else:
            thrust_scale = 2        
        # Because we have the toggle, we'll always initialize these
        q_t = ax.quiver(qt['x'],qt['y'],qt['dx'],qt['dy'], scale = thrust_scale, angles='xy', scale_units='xy', color = 'orange', pivot = 'tail', zorder = 4)
        # Changed the scale here to be more visible
        q_v = ax.quiver(qv['x'],qv['y'],qv['dx'],qv['dy'], scale= 1/50, angles='xy', scale_units='xy', color = 'pink', pivot = 'tail', zorder = 4)

    global current_time 
    current_time = 0.0

    def update(frame):
        global current_time
        # Always compute physics each frame
        Body.timestep(time_step = time_step)
        if mainship._find_target() is not None:
            rel_position, rel_velocity = mainship.compute_relative_to_target()
            relative_position_label.set_visible(settings.rel_stats_on)
        else:
            rel_position, rel_velocity = (0,0), (0,0)
            relative_position_label.set_visible(False)
        # Units don't really matter here since it's just for display
        
        relative_position_label.set_text(f'Distance to Target:\n'
                                         f'({rel_position[0]:.0f}, {rel_position[1]:00.0f}) m\n'
                                         f' |d|: {np.linalg.norm(rel_position):00.0f} m\n'
                                         f'Relative Velocity:\n'
                                         f'({rel_velocity[0]:.0f}, {rel_velocity[1]:.0f}) m/s\n'
                                         f'|dv|: {np.linalg.norm(rel_velocity):.0f} m/s')

        current_time += time_step
        if frame % 5 == 0: # Reduce speed for better visualization
            time_label.set_text(f'Time: {current_time:.0f} s')

        toggle_label.set_visible(settings.help_on)
        

        for i, path in enumerate(path_lines):
            path.set_data(Bodies[i].path[:,0],Bodies[i].path[:,1])

            if isinstance(Bodies[i], Spacecraft):
                path.set_visible(settings.ship_path_on)
                if Bodies[i].is_crashed:
                    path.set_color('red')
                    path.set_linestyle('--')
                else:
                    path.set_color(Bodies[i].color)
                    path.set_linestyle('-')
            else :
                path.set_visible(settings.planet_path_on)
                if Bodies[i].is_dynamically_updated == False:
                    path.set_data([],[])

        # Update vector field if any Bodies are dynamic
        q.set_visible(settings.vector_field_on)
        if settings.vector_field_on:
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
            shadow.set_visible(settings.potential_field_on)

        if Ships:
            qv,qt = body_vectors([mainship])
            q_v.set_visible(settings.vel_vector_on)
            q_v.set_UVC(qv['dx'], qv['dy'])
            q_v.set_offsets(np.array([[qv['x'], qv['y']]]))

            q_t.set_visible(settings.thrust_vector_on and mainship.thrust_mag != 0)
            q_t.set_UVC(qt['dx'], qt['dy'])
            q_t.set_offsets(np.array([[qt['x'], qt['y']]]))
        
        pot_artists = [*path_lines, *body_circles, *shadow_circles, q, follow_line, q_t, q_v, time_label, toggle_label, relative_position_label,title_label]  
        
        artists = [artist for artist in pot_artists if artist is not None]
            
        return artists
    
    ani = animation.FuncAnimation(fig, update, frames=100, interval=1, blit=True, repeat=True)
    # update(0) # Debug requires this to not run on GUI loop
    plt.show()

    # ani.save('flowertest.gif', dpi=100, writer='pillow')

    return ani, fig, ax

