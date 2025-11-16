from classes import Body, Spacecraft, Bounds

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

    if scenario == '1': # Goal and spaceship is in two different corner
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
                   velocity = (-1,1), 
                   thrust = 50.0, 
                   color = 'white', 
                   radius = 1, 
                   is_dynamically_updated = True)
        
        Spacecraft(name = 'target',    
                   mass = 0,  
                   position = (500,500), 
                   velocity = (0,0), 
                   color = 'purple', 
                   radius = 20, 
                   is_dynamically_updated = False,
                   is_target = True)

        bounds = Bounds(-600, 600, -600, 600)

    elif scenario == '2': # Orbiting ship and target, stationary planet
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

        bounds = Bounds(-600, 600, -600, 600)

    elif scenario == '3': # Orbiting moon, ship, and target
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
                   velocity = (-1.2,1.2), 
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

        bounds = Bounds(-600, 600, -600, 600)

    elif scenario == 'two_body': # Stable figure 8 2 body
        Body(
            name= 'star1', 
            mass = 500, 
            position = (-100,0), 
            velocity = (0,1.11803398875),  
            color = 'blue', 
            is_dynamically_updated = True)
        
        Body(name= 'star2', 
             mass = 500, 
             position = (100,0),  
             velocity = (0,-1.11803398875), 
             color = 'red',  
             is_dynamically_updated = True)
    
        Spacecraft(name ='spaceshipA', 
                   mass = 10, 
                   position = (0,0), 
                   velocity = (-1,1),
                   color = 'white',
                   thrust=10)
        
        Spacecraft(name ='target', 
                   mass = 10, 
                   position = (150,0), 
                   velocity = (0,1),
                   color = 'green',
                   radius = 5,
                   thrust=0.0,
                   is_target=True)

        bounds = Bounds(-150, 150, -150, 150)

    elif scenario == 'clover': # Clover 3 body
        Body(name= 'star1', 
             mass = 2000, 
             position = (0,0),       
             velocity = (0,0), 
             color = 'blue',   
             radius = 50, 
             is_dynamically_updated = False)
        Body(name= 'star2', 
             mass = 1500, 
             position = (-300,700),  
             velocity = (0,0), 
             color = 'red',    
             radius = 50, 
             is_dynamically_updated = False)
        Body(name= 'star3', 
             mass = 600,  
             position = (-520,-350), 
             velocity = (0,0), 
             color = 'yellow', 
             radius = 50, 
             is_dynamically_updated = False)

        Spacecraft(name ='spaceshipA', 
                   mass = 10, 
                   position = (0,-500),   
                   velocity = (-3,3), 
                   thrust = 50.0, 
                   color = 'white', 
                   radius = 10, 
                   is_dynamically_updated = True)
        
        Spacecraft(name = 'target',    
                   mass = 0,  
                   position = (-500,500), 
                   velocity = (0,0), 
                   color = 'purple', 
                   radius = 20, 
                   is_dynamically_updated = False,
                   is_target = True)

        bounds = Bounds(-600, 600, -600, 600)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return bounds