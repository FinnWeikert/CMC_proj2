

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
from util.rw import load_object
import matplotlib.pyplot as plt
from plotting_common import plot_2d


# Global parameters to define what code segments to run
GSS = True              # for figure 13 in pdf
IDIFF = True            # for figure 14 & 15 in pdf 
WAVEFREQ = True         # for figure 16 in pdf
SIMS_GSS = True         # for figure 17 in pdf
PAR_SEARCH = True       # for figure 18 in pdf

def exercise7():
    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")

    if GSS:
        # Number of simulations
        nsim = 30

        # Range of input I values
        I_list = np.linspace(0.05, 30, nsim)

        # Different values of w_stretch to test
        w_stretch_values = [0, 1, 5, 10, 15]

        # Function to run simulations for a given w_stretch value
        def run_simulations(w_stretch, I_list, nsim, log_path):
            # Prepare the list of simulation parameters
            pars_list = [
                SimulationParameters(
                    simulation_i=i,
                    n_iterations=7500,
                    I=I,
                    w_stretch=w_stretch,
                    log_path=log_path,
                    video_record=False,
                    compute_metrics=3,
                    headless=True,
                    print_metrics=False,
                    return_network=True
                )
                for i, I in enumerate(I_list)
            ]

            # Check if the simulation results already exist
            log_files = os.listdir(log_path)
            num_files = len(log_files)

            if num_files != nsim:
                controllers = run_multiple(pars_list, num_process=8)
            else:
                controllers = [load_object(f"{log_path}/controller{i}") for i in range(nsim)]

            # Extract the ptcc metrics
            ptcc_list = [controller.metrics['ptcc'] for controller in controllers]
            return ptcc_list

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        for w_stretch in w_stretch_values:
            ptcc_list = run_simulations(w_stretch, I_list, nsim, log_path)
            ax.plot(I_list, ptcc_list, linewidth=2, label=f'$g_{{ss}}={w_stretch}$')

        # Customize the plot
        ax.axhline(y=1.5, color='r', linestyle='--', label='stable oscillation limit')
        #ax.axvline(x=0, color='grey', linestyle='-')
        #ax.axvline(x=30, color='grey', linestyle='-')
        ax.set_xlabel('input I')
        ax.set_ylabel('ptcc')
        ax.set_title('Peak-to-trough Correlation as Function of I')
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_ylim(1, 2)


    if IDIFF:
        # Number of simulations
        nsim = 10

        # Range of differential drive I_diff values
        Idiff_list = np.linspace(0, 4, nsim)
        #Idiff_list = [1, 2, 3, 4]

        # Different values of g_ss to test
        g_ss_values = [0, 7.5, 15]

        # Function to run simulations for a given g_ss value
        def run_simulations(g_ss, Idiff_list, nsim, log_path):
            # Prepare the list of simulation parameters
            pars_list = [
                SimulationParameters(
                    simulation_i=i,
                    n_iterations=7501,
                    Idiff=Idiff,
                    w_stretch=g_ss,
                    log_path=log_path,
                    video_record=False,
                    compute_metrics=3,
                    headless=True,
                    print_metrics=False,
                    return_network=True
                )
                for i, Idiff in enumerate(Idiff_list)
            ]

            # Check if the simulation results already exist
            log_files = os.listdir(log_path)
            num_files = len(log_files)

            if num_files != 2*nsim:
                controllers = run_multiple(pars_list, num_process=8)
            else:
                controllers = [load_object(f"{log_path}/controller{i}") for i in range(nsim)]

            # Extract the metrics
            curvature_list = [controller.metrics['curvature'] for controller in controllers]
            lspeed_PCA_list = [controller.metrics['lspeed_PCA'] for controller in controllers]
            lspeed_cycle_list = [controller.metrics['lspeed_cycle'] for controller in controllers]

            return controllers, curvature_list, lspeed_PCA_list, lspeed_cycle_list

        # Plotting
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        for g_ss in g_ss_values:
            controllers, curvature_list, lspeed_PCA_list, lspeed_cycle_list = run_simulations(g_ss, Idiff_list, nsim, log_path)
            ax1.plot(Idiff_list, curvature_list, linewidth=2, label=f'$g_{{ss}}={g_ss}$')

        # Customize the plot
        ax1.set_xlabel('differential drive $I_{{diff}}$')
        ax1.set_ylabel('Curvature')
        ax1.set_title('Curvature as Function of $I_{{diff}}$')
        ax1.legend(fontsize=10)
        ax1.grid(True)

        fig1.tight_layout()
        plt.show()


    if SIMS_GSS:

        log_path = './logs/exercise7/multiple_sim/'
        os.makedirs(log_path, exist_ok=True)

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
                I = 20,
                compute_metrics=3, # changed
                headless=True,
                print_metrics=False,
                return_network=True
            )
            for i, gss in enumerate(gss_list)
        ]

        # check if this aprameter search was run before if so acces log
        log_controlers = os.listdir("logs/exercise7/multiple_sim/")
        # Count the number of file
        num_files = len(log_controlers)
        
        # if not corresponding number of simulations stored in logs, run the simulations
        if num_files != nsim:
            controllers = run_multiple(pars_list, num_process=8)
        else: # load the simulations from logs
            controllers = []
            for i in range(nsim):
                controllers.append(load_object("logs/exercise7/multiple_sim/controller"+str(i)))

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



    if PAR_SEARCH:
    
        log_path = './logs/exercise7/Fspeed/'
        os.makedirs(log_path, exist_ok=True)

        nsim = 10
        # Lists to store amplitudes and wave frequencies per sim
        I_list = []
        gss_list = []

        pylog.info(
            "Running multiple simulations in parallel from a list of SimulationParameters")
        pars_list = [
            SimulationParameters(
                simulation_i=i*nsim+j,
                n_iterations=5001,
                log_path=log_path,
                video_record=False,
                compute_metrics=3, # changed
                w_stretch=gss,
                I=I,
                headless=True,
                print_metrics=False,
                return_network=True # added
            )
            for i, I in enumerate(np.linspace(0.05, 30, nsim))
            for j, gss in enumerate(np.linspace(0, 15, nsim))
            for _ in (I_list.append(I), gss_list.append(gss))
        ][::2] # remove every second because the line above makes doubles the params otherwise

        # check if this aprameter search was run before if so acces log
        log_controlers = os.listdir("logs/exercise7/Fspeed/")
        # Count the number of files
        num_files = len(log_controlers)

        # if not corresponding number of simulations stored in logs, run the simulations
        if num_files != nsim**2:
            controllers = run_multiple(pars_list, num_process=8)
        else: # load the simulations from logs
            controllers = []
            for i in range(nsim**2):
                controllers.append(load_object("logs/exercise7/Fspeed/controller"+str(i)))

        # perform the parameter search
        # dict to store opti params
        optim_para = [0, None, None, None]  # [speed, opti_I, opti_gss, sim_index]

        # 2d array of dimension [N, 3], N = number of controllers
        # first col: amps, second: wavefreq, and last: speed
        para_search_results_PCA = np.zeros((len(controllers), 3))

        # parameter search for highest amp and wavefrequency with highest speed
        for i, controller in enumerate(controllers):

            # extract metrics for current controller
            I = I_list[controller.pars.simulation_i]
            gss = gss_list[controller.pars.simulation_i]
            fspeed_PCA = controller.metrics['fspeed_PCA']

            # store the parameters and corresponding speed in results array
            para_search_results_PCA[i][0] = I
            para_search_results_PCA[i][1] = gss
            para_search_results_PCA[i][2] = fspeed_PCA

            # fspeed PCA
            if fspeed_PCA > optim_para[0]:
                optim_para[0] = fspeed_PCA  # update fspeed
                optim_para[1] = I  # update opti I
                optim_para[2] = gss  # update opti gss
                optim_para[3] = controller.pars.simulation_i  # update simulation index

        # Print debug info
        d = 1  # debug

        print("An optimal Forward speed of", optim_para[0],"is reached for a stretch feedback strenght of", optim_para[2])
        # Plot the results
        # Plot the heat map of the parameter search (using plot2d)
        labels = ['Input drive', 'Feedback strength', 'Forward speed (PCA)']
        plt.figure('2D Parameter Search PCA Fspeed', figsize=[10, 10])
        plot_2d(para_search_results_PCA, labels, cmap='nipy_spectral')  # can maybe find nicer cmap='coolwarm' or other
        plt.title('2D Parameter Search PCA Fspeed')

if __name__ == '__main__':
    exercise7()
    plt.show()

