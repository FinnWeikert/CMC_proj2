

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
from plotting_common import plot_left_right, plot_time_histories_multiple_windows
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
import os
from util.rw import load_object

# READ: global parameters to defines what to run
SINGLE_SIM = False # single sim with metrics output
MULTIPLE_SIM = True # multiple sim with plots

def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    if SINGLE_SIM:
        # Run an individual simulations with default parameters
        all_pars = SimulationParameters(
            n_iterations=5001, # CHANGE to 10k at the end
            compute_metrics=3,
            w_stretch=7,
            return_network=True,
            headless=True,
            print_metrics=True
        )

        pylog.info("Running the simulation")
        controller = run_single(
            all_pars
        )

        pylog.info("Plotting the result")

        cutoff = 2000

        # muscle activities plot
        left_muscle_idx = controller.muscle_l
        right_muscle_idx = controller.muscle_r
        plt.figure('muscle_activities')
        plot_left_right(
            controller.times[:cutoff],
            controller.state[:cutoff],
            left_muscle_idx[:cutoff],
            right_muscle_idx[:cutoff],
            offset=0.75)
        
        # CPG activities plot
        left_CPG_idx = controller.rL
        right_CPG_idx = controller.rR
        plt.figure('CPG_activities')
        plot_left_right(
            controller.times[:cutoff],
            controller.state[:cutoff],
            left_CPG_idx[:cutoff],
            right_CPG_idx[:cutoff],
            offset=0.75)
        
        # sensory neurons plot 
        left_sens_idx = controller.sL
        right_sens_idx = controller.sR
        plt.figure('sensory_neurons_activities')
        plot_left_right(
            controller.times[:cutoff],
            controller.state[:cutoff],
            left_sens_idx[:cutoff],
            right_sens_idx[:cutoff],
            offset=0.75)

        jp = controller.joints_positions
        d=1
        # +++ need to do joint angle positions plot
        # pos = np.array(self.data.sensors.joints.positions(iteration)[4:-1]) => from controller.py
        plt.figure("joint positions_single")
        plot_time_histories_multiple_windows(
            controller.times,
            controller.joints_positions,
            offset=-0.4,
            colors="green",
            ylabel="joint positions",
            lw=1
        )
        # need to make this claener...


    if MULTIPLE_SIM:
        #Now vary gss ∈ [0,15], how does the frequency, wavefrequency and forward speed change?

        nsim = 24  # Number of samples
        gss_list = np.linspace(0, 15, nsim)

        pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
        pars_list = [
            SimulationParameters(
                simulation_i=i,
                n_iterations=7501,
                w_stretch = gss,
                log_path=log_path,
                video_record=False,
                compute_metrics=3, # changed
                headless=True,
                print_metrics=False,
                return_network=True
            )
            for i, gss in enumerate(gss_list)
        ]

        # check if this aprameter search was run before if so acces log
        log_controlers = os.listdir("logs/exercise6/")
        # Count the number of file
        num_files = len(log_controlers)
        
        # if not corresponding number of simulations stored in logs, run the simulations
        if num_files != nsim:
            controllers = run_multiple(pars_list, num_process=8)
        else: # load the simulations from logs
            controllers = []
            for i in range(nsim):
                controllers.append(load_object("logs/exercise6/controller"+str(i)))

        frequency_list = []
        wavefreq_list = []
        fspeed_PCA_list = []
        fspeed_cycle_list = []

        for i, controller in enumerate(controllers):
            frequency_list.append(controller.metrics['frequency'])
            wavefreq_list.append(controller.metrics['wavefrequency'])
            fspeed_PCA_list.append(controller.metrics['fspeed_PCA'])
            fspeed_cycle_list.append(controller.metrics['fspeed_cycle'])

        fig1 = plt.figure('Frequencies',figsize=(10, 6))
        plt.plot(gss_list, frequency_list, label='frequency', linewidth=1.5)
        plt.plot(gss_list, wavefreq_list, label='wavefrequency', linewidth=1.5)
        plt.xlabel('Stretch strength gss')
        plt.ylabel('[Hz]')
        plt.title('Frequency and Wavefrequency as Function of gss')
        plt.legend(fontsize=10)  # Adjust legend size
        plt.grid(True)

        fig2 = plt.figure('Fspeed', figsize=(10, 6))
        plt.plot(gss_list, fspeed_PCA_list, label='Fspeed PCA', linewidth=2.5)
        plt.plot(gss_list, fspeed_cycle_list, label='Fspeed cylce', linewidth=2.5)
        plt.xlabel('Stretch strength gss')
        plt.ylabel('Speed')
        plt.title('Forward speed as Function of gss')
        plt.legend(fontsize=10)  # Adjust legend size
        plt.grid(True)

        # EXO: Les plots sont un peu bizarres => demander si ca fait du sens?

if __name__ == '__main__':
    exercise6()
    plt.show()

