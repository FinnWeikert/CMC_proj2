

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
from util.rw import load_object
import matplotlib.pyplot as plt
from plotting_common import save_figure, plot_left_right, plot_2d

SINGLE_SIM = False
MULTIPLE_SIM = True
def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    log_path = './logs/exercise7/'
    os.makedirs(log_path, exist_ok=True)
    '''
    nsim = 30  # Number of samples
    gss_list = np.linspace(0, 15, nsim)
    I_list = np.linspace(0,30 , nsim)

    gsss= []
    Is = []
    
    pylog.info(
    "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=2500,
            w_stretch=gss,
            I=I,
            log_path=log_path,
            video_record=False,
            compute_metrics=3, # changed
            headless=True,
            print_metrics=False,
            return_network=True
        )
        for i,gss in enumerate(gss_list)
        for j, I in enumerate(I_list)
        for _ in (gsss.append(gss), Is.append(I))
    ][::2] # remove every second because the line above makes doubles the params otherwise
    

    # check if this parameter search was run before, if so: access log
    log_controllers = os.listdir("logs/exercise7/")
    num_files = len(log_controllers)

    # if not corresponding to number of sim stored
    if num_files != nsim:
        controllers = run_multiple(pars_list2, num_process=8)
    else:
        controllers = []
        for i in range(nsim):
            controllers.append(load_object("logs/exercise7/controller"+str(i)))

    frequency_list = []
    wavefrequency_list = []
    fspeed_PCA_list = []
    fspeed_cycle_list = []
    ptcc_list = []
    for i, controller in enumerate(controllers):
        frequency_list.append(controller.metrics['frequency'])
        wavefrequency_list.append(controller.metrics['wavefrequency'])
        fspeed_PCA_list.append(controller.metrics['fspeed_PCA'])
        fspeed_cycle_list.append(controller.metrics['fspeed_cycle'])
        ptcc_list.append(controller.metrics['ptcc'])

    # Create dictionnaries for 2D plots
    para_search_results_gss_I_ptcc = np.zeros((len(controllers), 3))
    for i in np.arange(len(controllers)):
        #para_search_results_gss_I_ptcc[i][0] = gsss[i]
        para_search_results_gss_I_ptcc[i][1] = Is[i]
        para_search_results_gss_I_ptcc[i][2] = ptcc_list[i]
    labelptcc = ['gss','I','Peak-to-through']
    # Plots
    
    plt.figure('2D parameter search ptcc', figsize=[10, 10])
    plot_2d(para_search_results_gss_I_ptcc,labelptcc, cmap='nipy_spectral')
    '''

    if MULTIPLE_SIM:
        nsim = 30  # Number of samples
        #base = 2  # Logarithmic base

        # steepnesses = np.logspace(np.log2(0.1), np.log2(100), nsim, base=base)
        I_list = np.linspace(0, 30, nsim)

        pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
        pars_list = [
            SimulationParameters(
                simulation_i=i,
                n_iterations=7500,
                I=I,
                w_stretch = 1,
                log_path=log_path,
                video_record=False,
                compute_metrics=3, # changed
                headless=True,
                print_metrics=False,
                return_network=True
            )
            for i, I in enumerate(I_list)
        ]

        # check if this aprameter search was run before if so acces log
        log_controlers = os.listdir("logs/exercise7/")
        # Count the number of file
        num_files = len(log_controlers)
        print(num_files)
        print(nsim)
        # if not corresponding number of simulations stored in logs, run the simulations
        if num_files != nsim:
            controllers = run_multiple(pars_list, num_process=8)
        else: # load the simulations from logs
            controllers = []
            for i in range(nsim):
                controllers.append(load_object("logs/exercise7/controller"+str(i)))

        ptcc_list = []
        for controller in controllers:
            ptcc_list.append(controller.metrics['ptcc'])

        fig1 = plt.figure('ptcc', figsize=(10, 6))
        plt.axhline(y=1.5, color='r', linestyle='--', label='stable oscillation limit')  # Add horizontal line at y=0.033
        plt.axvline(x=0, color='grey', linestyle='-')
        plt.axvline(x=30, color='grey', linestyle='-')
        plt.step(I_list, ptcc_list, linewidth=3)
        plt.xlabel('input I')
        plt.ylabel('ptcc')
        plt.title('Peak to through Correlation as Function of I')
        plt.legend(fontsize=10)  # Adjust legend size
        plt.grid(True)
        # Add shaded region between the two vertical lines
        plt.fill_betweenx(np.linspace(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], 100), 0, 30, color='lightgrey')
        # Add text annotations for x-values at the level of the x-axis
        plt.text(0, plt.gca().get_ylim()[0]-0.02, 'I=0', ha='left', va='top', color='grey')
        plt.text(30, plt.gca().get_ylim()[0]-0.02, 'I=30', ha='left', va='top', color='grey')
    #def main(ind=0, w_stretch=0):

    #log_path = './logs/exercise7/w_stretch'+str(ind)+'/'
    #os.makedirs(log_path, exist_ok=True)
        
if __name__ == '__main__':
    exercise7()
    plt.show()
