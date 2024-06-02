from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import numpy as np
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog


def exercise3(**kwargs):

    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")
    log_path = './logs/exercise3/'
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=10001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )

    pylog.info("Plotting the result")

    cutoff = 2000
    left_muscle_idx = controller.muscle_l
    right_muscle_idx = controller.muscle_r

    # example plot using plot_left_right
    plt.figure('muscle_activities')
    plot_left_right(
        controller.times[:cutoff],
        controller.state[:cutoff],
        left_muscle_idx,
        right_muscle_idx,
        offset=1)
    
    left_CPG_idx = controller.rL
    right_CPG_idx = controller.rR

    plt.figure('CPG_activities')
    plot_left_right(
        controller.times[:cutoff],
        controller.state[:cutoff],
        left_CPG_idx,
        right_CPG_idx,
        offset=0.8)

if __name__ == '__main__':
    exercise3(headless=True) # should be True in the end!
    plt.show()
