

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
from plotting_common import plot_left_right, plot_time_histories_multiple_windows
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
import os
from util.rw import load_object

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
            n_iterations=10001, # CHANGE to 10k at the end
            compute_metrics=3,
            w_stretch=4,
            return_network=True,
            headless=False,
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
            left_muscle_idx,
            right_muscle_idx,
            offset=0.8)
        
        # CPG activities plot
        left_CPG_idx = controller.rL
        right_CPG_idx = controller.rR
        plt.figure('CPG_activities')
        plot_left_right(
            controller.times[:cutoff],
            controller.state[:cutoff],
            left_CPG_idx,
            right_CPG_idx,
            offset=0.8)
        
        # sensory neurons plot 
        left_sens_idx = controller.sL
        right_sens_idx = controller.sR
        plt.figure('sensory_neurons_activities')
        plot_left_right(
            controller.times[:cutoff],
            controller.state[:cutoff],
            left_sens_idx,
            right_sens_idx,
            offset=0.3)

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, 15)]
        joint_labels = ["joint " + str(i+1) for i in range(15)]
        plt.figure("joint positions_single")
        plot_time_histories_multiple_windows(
            controller.times[:cutoff],
            controller.joints_positions[:cutoff],
            colors=colors,
            ylabel="joint positions",
            labels=joint_labels,
            lw=1
        )


    if MULTIPLE_SIM:
        #Now vary gss ∈ [0,15], how does the frequency, wavefrequency and forward speed change?

        nsim = 40  # Number of samples
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
        amplitude_list = []
        fspeed_PCA_list = []
        fspeed_cycle_list = []

        for i, controller in enumerate(controllers):
            frequency_list.append(controller.metrics['frequency'])
            wavefreq_list.append(controller.metrics['wavefrequency'])
            amplitude_list.append(controller.metrics['amp'])
            fspeed_PCA_list.append(controller.metrics['fspeed_PCA'])
            fspeed_cycle_list.append(controller.metrics['fspeed_cycle'])


        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Plot frequency on primary y-axis (left)
        color = 'tab:blue'
        ax1.set_xlabel('Stretch strength gss')
        ax1.set_ylabel('Freqeuncy [Hz]', color=color)
        ax1.plot(gss_list, frequency_list, color=color, label='frequency', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a secondary y-axis and plot lspeed on it (right)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Wavefrequency [Hz]', color=color)
        ax2.plot(gss_list, wavefreq_list, color=color, label='wavefrequency', linewidth=2)  
        ax2.tick_params(axis='y', labelcolor=color)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=13)

        fig1.tight_layout()  
        plt.title('Frequency and Wavefrequency as Function of gss')
        plt.grid(True)

        fig2 = plt.figure('Fspeed', figsize=(10, 6))
        plt.plot(gss_list, fspeed_PCA_list, label='Fspeed PCA', linewidth=2.5)
        #plt.plot(gss_list, fspeed_cycle_list, label='Fspeed cylce', linewidth=2.5)
        plt.xlabel('Stretch strength gss')
        plt.ylabel('Speed')
        plt.title('Forward speed as Function of gss')
        plt.legend(fontsize=10)  # Adjust legend size
        plt.grid(True)

        # EXO: Les plots sont un peu bizarres => demander si ca fait du sens?

if __name__ == '__main__':
    exercise6()
    plt.show()

