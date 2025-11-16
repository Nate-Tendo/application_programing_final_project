# 2D Space Navigation ME 369 Final Project

## Collaborators
Nathan Lee, Isaac Sorensen, Philip Bortolotti


## Phases
0. Initialize Universe with Spaceship w/ Initial Conditions -- By 11/8 (Sat)
1. Rocket through stationary universe with stationary target -- By 11/8 (Sat)
2. Rocket through moving universe with stationary target -- By 11/10 (Mon)
3. Rocket through moving universe with moving target -- By 11/12 (Wed)
4. Pursuit (Real-Time Prediction) -- After Thurs 11/13

## Task List

1. Create a version control procedure (Isaac)
2. Expand/Create Plotting Function (Nate)
3. Update Vector Class
4. Update Body Class
5. Add a net-force vector to the spaceship
6. Body.Update()
7. 

simulator:

t = 1.56
spaceship.update()
bodies.update()


A. How is our animation process going to work? How will we step through time?
B. How will we plot vector fields?
C. How will we numerically solve the spaceship path?
D. How will we implement thrust? (Angle and magnitude)
D. Optimization
    i. Make cost function
    ii. Parameterize path
    iii. Update cost function stuff


Things to add:
1. Ability to turn on/off trails for all bodies
2. Ability to set whether a body is stationary or not
3. Ability to turn on/off vector field
4. Ability to set the animation for a certain number of seconds OR end a certain amount of time after the spaceship is no longer on screen
5. Spaceship be a triangle that rotates properly with velocity vector
6. Redefine functions such that all bodies will be stepped automatically, without having to call each one, while also skipping stationary objects
7. Ability to plot lagrange points
8. Preset configurations for demo purposes such as
    a. Binary star configurations
    b. Stable solutions to the three body problem
    c. Solar systems (perhaps with predefined orbits such that we do not compute them)
9. Hyperbolic trajectory
10. Have lines glow, fade out


TO do
1. Straight line path, how much fuel does it cost?
    a. for ANY GIVEN PATH, how much fuel does it use
2. Gravity field to decide path
3. Have a naive path where it doesn't predict (presimulate), only respond
3. Implement a way to presimulate and then do the optimal path
    a) Static single planet
    b) solar system
    c) binary figure 8 or three body system


Tasks:
1. Fix order of stepping to conserve momentum
2. Create a pathfinding algorithm - Make it follow a path, then add a PID, then have a
3. Functions to create preset configurations for the demo

Add vectors for thrust and velocity

Visualization of 3D spacetime gravity over time - topology

Normalize all arrows based on what looks good on the plot

imshow heatmap

Note to self: after presentation is done refactor code to have a list of all time of Bodies class of the position, velocity, acceleration, thrust, etc of every body



Look into Numba JIT just in time later on