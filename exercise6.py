

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
from plotting_common import plot_left_right
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
import os

# READ: global parameters to defines what to run
SINGLE_SIM = True # single sim with metrics output
MULTIPLE_SIM = False # multiple sim with plots

def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    if SINGLE_SIM:
        # Run an individual simulations with default parameters
        all_pars = SimulationParameters(
            n_iterations=10001,
            compute_metrics=3,
            w_stretch=5,
            return_network=True,
            headless=False,
            print_metrics=True
        )

        pylog.info("Running the simulation")
        controller = run_single(
            all_pars
        )

        pylog.info("Plotting the result")

        # muscle activities plot
        left_muscle_idx = controller.muscle_l
        right_muscle_idx = controller.muscle_r
        plt.figure('muscle_activities')
        plot_left_right(
            controller.times,
            controller.state,
            left_muscle_idx,
            right_muscle_idx,
            cm="green",
            offset=0.1)
        
        # CPG activities plot
        left_CPG_idx = controller.rL
        right_CPG_idx = controller.rR
        plt.figure('CPG_activities')
        plot_left_right(
            controller.times,
            controller.state,
            left_CPG_idx,
            right_CPG_idx,
            cm="green",
            offset=0.1)
        
        # sensory neurons plot 
        left_sens_idx = controller.sL
        right_sens_idx = controller.sR
        plt.figure('sensory_neurons_activities')
        plot_left_right(
            controller.times,
            controller.state,
            left_sens_idx,
            right_sens_idx,
            cm="green",
            offset=0.1)

        # +++ need to do joint angle positions plot

    if MULTIPLE_SIM:
        #Now vary gss ∈ [0,15], how does the frequency, wavefrequency and forward speed change?
        d = 1
if __name__ == '__main__':
    exercise6()
    plt.show()

