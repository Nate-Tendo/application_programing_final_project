from classes import Vector2, Body, Spacecraft

def initialize_scenario(scenario_name: str):
    """Initializes bodies and spacecraft based on the selected scenario."""
    Body._instances.clear()
    Spacecraft._instances.clear()
    
    if scenario_name == "solar_system":
        # Example: Simple solar system with one star and two planets
        star = Body(name='Sun', mass=1.989e30, position=Vector2(0, 0), velocity=Vector2(0, 0), color='yellow', radius=50)
        planet1 = Body(name='Earth', mass=5.972e24, position=Vector2(1.496e11, 0), velocity=Vector2(0, 29780), color='blue', radius=10)
        planet2 = Body(name='Mars', mass=6.39e23, position=Vector2(2.279e11, 0), velocity=Vector2(0, 24070), color='red', radius=8)
        spacecraft = Spacecraft(name='Voyager 1', mass=721.9, position=Vector2(1.496e11 + 4.216e7, 0), velocity=Vector2(0, 29780 + 17000), color='white', radius=5)
    
    elif scenario_name == "binary_star":
        # Example: Binary star system with a spacecraft
        star1 = Body(name='Star A', mass=2e30, position=Vector2(-5e10, 0), velocity=Vector2(0, -15000), color='orange', radius=40)
        star2 = Body(name='Star B', mass=2e30, position=Vector2(5e10, 0), velocity=Vector2(0, 15000), color='red', radius=40)
        spacecraft = Spacecraft(name='Explorer', mass=1000, position=Vector2(0, -1e11), velocity=Vector2(20000, 0), color='white', radius=5)

    elif scenario_name == "scenario_1":
        star1 = Body(name= 'star1', mass = 2000, position = Vector2(0,0), velocity = Vector2(0,0),color = 'blue', radius = 50)
        # star2 = Body(name= 'star2', mass = 1500, position = Vector2(-300,700), velocity = Vector2(0,0), color = 'red', radius = 50)
        # star3 = Body(name= 'star2', mass = 600, position = Vector2(-520,-350), velocity = Vector2(0,0), color = 'yellow', radius = 50)
        
        spaceshipA = Spacecraft(name ='spaceshipA', mass = 10, position = Vector2(0,-500), velocity = Vector2(0,0), thrust = 10.0, color = 'white', radius = 10)
        # target = Spacecraft(name = 'target', mass = 0, position = Vector2(-500,500), velocity = Vector2(0,0), color = 'purple', radius = 20 )
        
    else:
        raise ValueError(f"Scenario '{scenario_name}' is not defined.")
    
    return Body._instances, Spacecraft._instances