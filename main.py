from solar_system_config import initialize_universe
from visualization import plot_universe_animation
        
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
    # navigationStrategy = 'potential_field'
    # navigationStrategy = 'lyapunov_nonlinear'
    # navigationStrategy = 'lyapunov_pd'
    navigationStrategy = 'chase'
    # navigationStrategy = '_'

    # 3. Plotting Options and Start Animation
    time_step = .5
    ani, fig, ax = plot_universe_animation(Bodies, Ships, scenario_bounds, time_step, navigationStrategy)
    # =============================================================================================================