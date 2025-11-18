import numpy as np
from classes import Body, Spacecraft, GRAVITY_CONSTANT
from utils import Bounds

def initialize_universe(scenario: str):
    """
    Initialize the solar system configuration based on the selected scenario.

    Parameters
    ----------
    scenario : str
        The scenario to initialize. Options are 'solar_system', 'earth_moon', 'custom'.

    Returns
    -------
    None
    """
    # Clear existing bodies
    Body._instances.clear()
    Spacecraft._instances.clear()
    match scenario:
        case '1': # Goal and spaceship is in two different corner
            Body(name= 'planet1', mass = 2000, position = (2,0), velocity = (2,0), 
                 color = 'green', radius = 50, is_dynamically_updated = False)
            
            Spacecraft(name ='spaceshipA', mass = 1, position = (-500,-500), velocity = (-1,1), thrust = 50.0, 
                       color = 'white', radius = 20, is_dynamically_updated = True)
            
            Spacecraft(name = 'target',    
                       mass = 0,  
                       position = (500,500), 
                       velocity = (0,0), 
                       color = 'purple', 
                       radius = 20, 
                       is_dynamically_updated = False,
                       is_target = True)
    
            scenario_bounds = Bounds(-600, 600, -600, 600)

        case '2': # Orbiting ship and target, stationary planet
            Body(name= 'planet1', 
                 mass = 2000, 
                 position = (0,0),       
                 velocity = (0,0), 
                 color = 'green',   
                 radius = 50, 
                 is_dynamically_updated = False)
    
            Spacecraft(name ='spaceshipA', 
                       mass = 1, 
                       position = (-500,-500),   
                       velocity = (-1.2,1.2), 
                       thrust = 50.0, 
                       color = 'white', 
                       radius = 10, 
                       is_dynamically_updated = True)
            
            Spacecraft(name = 'target',    
                       mass = 0,  
                       position = (300,300), 
                       velocity = (1.2,-1.2), 
                       color = 'purple', 
                       radius = 20, 
                       is_dynamically_updated = True,
                       is_target = True)
    
            scenario_bounds = Bounds(-600, 600, -600, 600)

        case '3': # Orbiting moon, ship, and target
            Body(name= 'planet1', 
                 mass = 2000, 
                 position = (0,0),       
                 velocity = (0,0), 
                 color = 'green',   
                 radius = 50, 
                 is_dynamically_updated = False)
            
            Body(name= 'moon', 
                 mass = 150, 
                 position = (-500,500),  
                 velocity = (1.2,1.2), 
                 color = 'red',    
                 radius = 50, 
                 is_dynamically_updated = True)
    
            Spacecraft(name ='spaceshipA', 
                       mass = 1, 
                       position = (-500,-500),   
                       velocity = (0,0), 
                       thrust = 50.0, 
                       color = 'white', 
                       radius = 10, 
                       is_dynamically_updated = True)
            
            Spacecraft(name = 'target',    
                       mass = 0,  
                       position = (500,500), 
                       velocity = (1.2,-1.2), 
                       color = 'purple', 
                       radius = 20, 
                       is_dynamically_updated = True,
                       is_target = True)
    
            scenario_bounds = Bounds(-600, 600, -600, 600)

        case '2b_figure8': # Stable figure 8 2 body
            Body(
                name= 'star1', 
                mass = 2000, 
                position = (-200,0), 
                velocity = (0,1.11803398875),  
                color = 'blue', 
                radius = 20,
                is_dynamically_updated = True)
            
            Body(name= 'star2', 
                 mass = 2000, 
                 position = (200,0),  
                 velocity = (0,-1.11803398875), 
                 color = 'red',
                 radius = 20,
                 is_dynamically_updated = True)
        
            Spacecraft(name ='spaceshipA', # gravity helping
                       mass = 10, 
                       position = (-250,-250), 
                       velocity = (2,1),
                       radius = 10,
                       color = 'white',
                       thrust=1)
            
            Spacecraft(name ='target', 
                       mass = 0, 
                       position = (-300,220),
                       velocity = (0,0),
                       color = 'powderblue',
                       radius = 10,
                       thrust=0.0,
                       is_dynamically_updated = False,
                       is_target=True)
    
            scenario_bounds = Bounds(-300, 300, -300, 300)

        case '2b_figure8_chase': # Stable figure 8 2 body
            Body(name= 'star1', 
                mass = 1000, 
                position = (-200,0), 
                velocity = (0, 1.11803398875),  
                color = 'blue', 
                radius = 20,
                is_dynamically_updated = True)
            
            Body(name= 'star2', 
                 mass = 1000, 
                 position = (200, 0),  
                 velocity = (0,-1.11803398875), 
                 color = 'red',
                 radius = 20,
                 is_dynamically_updated = True)
        
            Spacecraft(name ='spaceshipA', # gravity helping
                       mass = 10.0, 
                       position = (0,0), 
                       velocity = (0,0),
                       radius = 2,
                       color = 'white',
                       thrust= 20.0)
            
            Spacecraft(name ='target', 
                       mass = 10, 
                       position = (10,0),
                       velocity = (1.1,0),
                       color = 'green',
                       radius = 5,
                       thrust= 0.0,
                       is_dynamically_updated = True,
                       is_target=True)
    
            scenario_bounds = Bounds(-300, 300, -300, 300)

        case '3b_figure8': # Figure-eight 3 body
            threebody_figeight(2000, 400)
            scenario_bounds = Bounds(-600, 600, -600, 600)
        
        case '3b_flower': # Flower shape 3 body
            threebody_flower(2000, 500)
            scenario_bounds = Bounds(-600, 600, -600, 600)

    return scenario_bounds, Body._instances, Spacecraft._instances

def onebody_circular_orbit(r,G,m):
    v = np.sqrt(G*m/r)
    pass

def sun_earth_moon_system():
    pass

def twobody_figeight():
    pass

def threebody_figeight(m,D):
    # Scale these dimensionless distances
    r1_dim = np.array([-0.97000436,  0.24308753])
    r2_dim = np.array([ 0.97000436, -0.24308753])
    r3_dim = np.array([ 0.0,         0.0       ])
    
    v1_dim = np.array([ 0.4662036850,  0.4323657300])
    v2_dim = np.array([ 0.4662036850,  0.4323657300])
    v3_dim = np.array([-0.93240737,   -0.86473146 ])
    
    scale = np.sqrt(GRAVITY_CONSTANT*m/D)
    
    r1 = D*r1_dim
    r2 = D*r2_dim
    r3 = D*r3_dim
    
    v1 = scale*v1_dim
    v2 = scale*v2_dim
    v3 = scale*v3_dim
    
    Body(name= 'planet1', 
         mass = m, 
         position = tuple(r1),       
         velocity = tuple(v1), 
         color = '#c1440e',   
         radius = 50, 
         is_dynamically_updated = True)
    Body(name= 'planet2', 
         mass = m, 
         position = tuple(r2),       
         velocity = tuple(v2), 
         color = '#d1e7e7',   
         radius = 50, 
         is_dynamically_updated = True)
    Body(name= 'planet3', 
         mass = m, 
         position = tuple(r3),       
         velocity = tuple(v3), 
         color = '#3f54ba',   
         radius = 50, 
         is_dynamically_updated = True)
    
    Spacecraft(name ='spaceshipA', # gravity helping
            mass = 0, 
            position = (-99999,0), 
            velocity = (0,0),
            radius = 0,
            color = 'white',
            thrust= 0,
            is_dynamically_updated= False)
    
def threebody_flower(m,D):
    r_dims = np.array([(0.0132604844,0), (1.4157286016,0), (-1.4289890859,0)])
    v_dims = np.array([(0,1.054151921), (0,-0.2101466639), (0,-0.8440052572)])
    
    r1_dim, r2_dim, r3_dim = r_dims
    v1_dim, v2_dim, v3_dim = v_dims
       
    scale = np.sqrt(GRAVITY_CONSTANT*m/D)
    
    r1 = D*r1_dim
    r2 = D*r2_dim
    r3 = D*r3_dim
    
    v1 = scale*v1_dim
    v2 = scale*v2_dim
    v3 = scale*v3_dim
    
    Body(name= 'planet1', 
         mass = m, 
         position = tuple(r1),       
         velocity = tuple(v1), 
         color = '#c1440e',   
         radius = 50, 
         is_dynamically_updated = True)
    Body(name= 'planet2', 
         mass = m, 
         position = tuple(r2),       
         velocity = tuple(v2), 
         color = '#d1e7e7',   
         radius = 50, 
         is_dynamically_updated = True)
    Body(name= 'planet3', 
         mass = m, 
         position = tuple(r3),       
         velocity = tuple(v3), 
         color = '#3f54ba',   
         radius = 50, 
         is_dynamically_updated = True)
    
    Spacecraft(name ='spaceshipA', # gravity helping
        mass = 0, 
        position = (-99999,0), 
        velocity = (0,0),
        radius = 0,
        color = 'white',
        thrust= 0,
        is_dynamically_updated= False)
