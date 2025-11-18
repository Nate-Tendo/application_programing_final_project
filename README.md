# 2D Space Navigation — ME 369 Final Project

## Presentation Link
https://docs.google.com/presentation/d/1GhS1o2qaUNRm4EfPN_qHtuG6O0BPAxVwPtxOwcx9kA8/edit?usp=sharing

---

# Overview

This project is a 2D spacecraft navigation simulator developed for the ME 369 Application Programming final project.

Our package simulates a spacecraft moving through a fabricated 2D “universe” containing gravitationally interacting celestial bodies. It demonstrates numerical physics, control strategies, and visualization techniques commonly used in orbital simulation.

Core goals:
- Universe Generation – Create dynamic environments with multiple gravitational bodies.
- Spacecraft Simulation – Model a thrust-capable ship orbiting and maneuvering in this universe.
- Navigation Strategies – Implement varied control algorithms (PID stay-put, line-following, potential fields, chase logic, etc.).
- Visualization Tools – Render trajectories, vector fields, and motion paths using matplotlib.

---

# Installation and Run Instructions

You will need:
- Python 3.x
- A terminal (macOS/Linux Terminal, Git Bash/WSL on Windows)
- Python libraries:
  - numpy
  - matplotlib

### 1. Clone the repository

    git clone https://github.com/Nate-Tendo/application_programing_final_project.git
    cd application_programing_final_project

### 2. (Optional) Create a virtual environment

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Windows:

    python -m venv .venv
    source .venv/Scripts/activate

### 3. Install dependencies

    pip install numpy matplotlib

### 4. Run the simulation

    python main.py

This will:
- Load universe and spacecraft settings from solar_system_config.py
- Run the configured simulation loop
- Visualize the resulting trajectories through visualization.py

---

# Simulation Settings

Simulation parameters are controlled through solar_system_config.py, including:

- Number and type of celestial bodies
- Initial positions, velocities, and masses
- Spacecraft initial state
- Time-step size and simulation duration
- The control strategy used in the run

### Editing the Universe

To alter simulation conditions (add planets, change gravity strength, reposition the ship, adjust scale), modify the dictionary values and class initializations in:

    solar_system_config.py

### Dynamics & Controllers

Dynamics are calculated using classical Newtonian physics and numerical integration (implemented in classes.py and utils.py).

Navigation/control algorithms include:
- PID Stay-Put
- Line-Following
- Potential Fields Path Solver
- Chase / Pursuit Algorithms
- Momentum Dampening
- Step-Decision Algorithms

Switching controllers is done inside main.py.

### Visualization Settings

visualization.py allows toggling (within the script) features such as:
- Path trails
- Vector field overlays
- Animation speed
- Axis scaling

---

# Keybinds

Control Toggles:

**H** – Help Menu  
**T** – Toggle Thrust Vector  
**V** – Toggle Velocity Vector  
**G** – Toggle Gravity Fields  
**F** – Toggle Potential Fields  
**P** – Toggle Ship Paths  
**L** – Toggle Planet Paths  
**R** – Reset Simulation  


---

# Collaborators

- Nathan Lee
- Isaac Sorensen
- Philip Bortolotti

---

# Dependencies

Python Libraries:
- numpy — numerical computations (gravity, integration, vector math).
- matplotlib — plots and animations.

Install via:

    pip install numpy matplotlib

No external tools (make, cmake, etc.) are required.

---

# File Structure

    ./
    ├── .gitignore              # Git ignore rules
    ├── LICENSE                 # MIT license
    ├── README.md               # This documentation
    ├── classes.py              # Simulation classes (bodies, spacecraft, controllers)
    ├── main.py                 # Entry point (runs a selected scenario)
    ├── solar_system_config.py  # Universe & configuration settings
    ├── utils.py                # Math helpers, physics utilities
    └── visualization.py        # Plotting and animation of trajectories

How the program flows:
1. solar_system_config.py sets up the universe + ship.
2. main.py loads the configuration, initializes the classes, and runs the simulation loop.
3. Physics & control logic are computed in classes.py and utils.py.
4. Final positions and trajectories are visualized via visualization.py.

---

# Acknowledgments

- Special thanks to the ME 369 teaching team for guidance.
- Thanks to contributors of open-source Python packages (numpy, matplotlib).

Use of AI Tools:
AI (ChatGPT) was used for debugging assistance, problem explanation, and generation of small helper functions or boilerplate code.
All AI-generated content was verified, edited, and validated by project members.

---

# Distribution of Work

| Member   | Contributions                                                                 |
|----------|-------------------------------------------------------------------------------|
| Philip   | Dynamic Solver, Vector Field Visualization, PID Stay-Put and Line-Following Algorithms |
| Nathan   | Class Structuring, Git Guru, Potential Fields Path Solver, Momentum Control Algorithms |
| Isaac    | Gravity Physics, UX, Visualization, Chase Algorithms, Step-Decision Algorithm, README documentation |

---

# Simplifying Assumptions

- The universe is empty except for our objects
- All objects are point masses
- Newtonian orbital mechanics are used
- Collisions are ignored
- All orbits used for navigation are stable
- The spaceship’s gravity does not affect other bodies
- The spaceship can apply thrust in any direction at any time
- The spaceship’s mass does not change with fuel usage

These assumptions simplify computation and keep the focus on navigation rather than astrophysical realism.

---

# License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

# About

This project showcases:
- 2D gravitational physics
- Orbital simulation
- Autonomous navigation strategies
- Data visualization and animation

Created as the final project for ME 369 Application Programming.
