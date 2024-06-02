
from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
import numpy as np
import farms_pylog as pylog
import os
from util.rw import load_object
import matplotlib.pyplot as plt
from plotting_common import plot_trajectory, plot_left_right, plot_time_histories, plot_time_histories_multiple_windows


# READ: global parameters to defines what to run
SINGLE_SIM = True # single sim with metrics output
MULTIPLE_SIM = False # multiple sim with plots

def exercise5():

    pylog.info("Ex 5")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    if SINGLE_SIM:
        # Run an individual simulations with default parameters
        all_pars = SimulationParameters(
            n_iterations=10001,
            compute_metrics=3,
            return_network=True,
            headless=True,
            print_metrics=True
        )

        pylog.info("Running the simulation")
        controller = run_single(
            all_pars
        )
        
        plt.figure("trajectory_single")
        plot_trajectory(controller, sim_fraction=0.2)

        # Note: for some reason need headless=False for this to plot correctly!
        cutoff = 2000
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

        plt.figure("link y-velocities_single")
        plot_time_histories(
            controller.times[:cutoff],
            controller.links_velocities[:cutoff, :cutoff, 1],
            offset=-0.,
            #colors="green",
            ylabel="link y-velocities",
            lw=1
        )


    if MULTIPLE_SIM:

        nsim = 10  # Number of samples
        Idiff_list = np.linspace(0, 4, nsim)

        pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
        pars_list = [
            SimulationParameters(
                simulation_i=i,
                n_iterations=7501,
                Idiff=Idiff,
                log_path=log_path,
                video_record=False,
                compute_metrics=3, # changed
                headless=True,
                print_metrics=False,
                return_network=True
            )
            for i, Idiff in enumerate(Idiff_list)
        ]

        # check if this aprameter search was run before if so acces log
        log_controlers = os.listdir("logs/exercise5/")
        # Count the number of file
        num_files = len(log_controlers)
        
        # if not corresponding number of simulations stored in logs, run the simulations
        if num_files != nsim:
            controllers = run_multiple(pars_list, num_process=8)
        else: # load the simulations from logs
            controllers = []
            for i in range(nsim):
                controllers.append(load_object("logs/exercise5/controller"+str(i)))

        curvature_list = []
        lspeed_PCA_list = []
        lspeed_cycle_list = []
        for i, controller in enumerate(controllers):
            curvature_list.append(controller.metrics['curvature'])
            lspeed_PCA_list.append(controller.metrics['lspeed_PCA'])
            lspeed_cycle_list.append(controller.metrics['lspeed_cycle'])

        # the plot is not pretty yet but this should be the result
        # QUESTION EXO: should we plot this?
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Plot curvature on primary y-axis (left)
        color = 'tab:blue'
        ax1.set_xlabel('differential drive I_diff')
        ax1.set_ylabel('Curvature', color=color)
        ax1.plot(Idiff_list, curvature_list, color=color, label='Curvature', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a secondary y-axis and plot lspeed on it (right)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Lateral Speed', color=color)
        ax2.plot(Idiff_list, lspeed_PCA_list, color=color, label='Lspeed_PCA', linewidth=2)
        ax2.plot(Idiff_list, lspeed_cycle_list, color='orange', label='Lspeed_Cycle', linewidth=2)  # Adding lspeed_cycle
        ax2.tick_params(axis='y', labelcolor=color)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

        fig1.tight_layout()  
        plt.title('Curvature and Lateral Speed as Function of I_diff')
        plt.grid(True)
        #plt.show()

        # ====================================================
        # Check that the turning radius match the curvature expected from the trajectory 
        # (i.e. plot the center of mass trajectory and the compute the radius=1/curvature).
        
        # Define the controllers to use
        controller_indices = [2, 4, 6, 8]

        # Create a figure with 2x2 subplots
        fig2, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Iterate over each controller index and subplot axes
        for idx, ax in zip(controller_indices, axs.flat):
            # Calculate center of mass positions for the current controller
            com_pos = np.mean(controllers[idx].links_positions[:-1, :, :2], axis=1)
            ax.plot(com_pos[:-1, 0], com_pos[:-1, 1], label="Center of Mass Positions")

            # Compute the radius for the current controller
            radius = 1 / np.abs(curvature_list[idx])  # Assuming curvature_list is defined

            # Add text to the plot (top right corner)
            ax.text(
                0, ax.get_ylim()[1] * 0.98,
                f"Computed Radius = {radius:.3f} m", fontsize=11, color='blue'
            )

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.axis('equal')
            ax.grid(True)

        # Adjust layout
        fig2.tight_layout()

        #plt.show()


        # ===============================================
        # Plot a swimming trajectory and neuron activities for fixed Idiff = 0 and Idiff > 0. 
        # How does the activities of left and right CPGs and MCs change for the case Idiff > 0?

        ### For Idiff = 0: 
        left_muscle_idx = controllers[0].muscle_l
        right_muscle_idx = controllers[0].muscle_r

        # example plot using plot_left_right
        plt.figure('muscle_activities')
        plot_left_right(
            controllers[0].times,
            controllers[0].state,
            left_muscle_idx,
            right_muscle_idx,
            cm="green",
            offset=0.1)
        
        left_CPG_idx = controllers[0].rL
        right_CPG_idx = controllers[0].rR

        plt.figure('CPG_activities')
        plot_left_right(
            controllers[0].times,
            controllers[0].state,
            left_CPG_idx,
            right_CPG_idx,
            cm="green",
            offset=0.1)

        ### For Idiff > 0:
        idx = 6

        left_muscle_idx = controllers[idx].muscle_l
        right_muscle_idx = controllers[idx].muscle_r

        # example plot using plot_left_right
        plt.figure('muscle_activities_Idiff')
        plot_left_right(
            controllers[idx].times,
            controllers[idx].state,
            left_muscle_idx,
            right_muscle_idx,
            cm="green",
            offset=0.1)
        
        left_CPG_idx = controllers[idx].rL
        right_CPG_idx = controllers[idx].rR

        plt.figure('CPG_activities_Idiff')
        plot_left_right(
            controllers[idx].times,
            controllers[idx].state,
            left_CPG_idx,
            right_CPG_idx,
            cm="green",
            offset=0.1)
        

# Mieux si on ajoute savefig...

if __name__ == '__main__':
    exercise5()
    plt.show()
