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

    if scenario == 'two_body':
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

    elif scenario == 'isaac_test1':
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