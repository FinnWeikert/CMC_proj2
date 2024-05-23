
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog
from util.rw import load_object
import matplotlib.pyplot as plt
from plotting_common import plot_2d


def exercise8():

    pylog.info("Ex 8")
    pylog.info("Implement exercise 8")
    log_path = './logs/exercise8/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 10
    # Lists to store amplitudes and wave frequencies per sim
    sigma_list = []
    gss_list = []

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=5001, # maybe this should be a bit larger to make sure intitial cond effect vanish
            method="noise",
            log_path=log_path,
            video_record=False,
            compute_metrics=3, # changed
            w_stretch=gss,
            noise_sigma=sigma,
            headless=True,
            print_metrics=False,
            return_network=True # added
        )
        for i, sigma in enumerate(np.linspace(0, 30, nsim))
        for j, gss in enumerate(np.linspace(0, 10, nsim))
        for _ in (sigma_list.append(sigma), gss_list.append(gss))
    ][::2] # remove every second because the line above makes doubles the params otherwise

    # check if this aprameter search was run before if so acces log
    log_controlers = os.listdir("logs/exercise8/")
    # Count the number of files
    num_files = len(log_controlers)

    # if not corresponding number of simulations stored in logs, run the simulations
    if num_files != nsim**2:
        controllers = run_multiple(pars_list, num_process=8)
    else: # load the simulations from logs
        controllers = []
        for i in range(nsim**2):
            controllers.append(load_object("logs/exercise8/controller"+str(i)))

    # perform the parameter search
    # dict to store opti params
    optim_para = [0, None, None, None]  # [speed, opti_sigma, opti_gss, sim_index]


    # 2d array of dimension [N, 3], N = number of controllers
    # first col: amps, second: wavefreq, and last: speed
    para_search_results_PCA = np.zeros((len(controllers), 3))

    # parameter search for highest amp and wavefrequency with highest speed
    for i, controller in enumerate(controllers):

        # extract metrics for current controller
        sigma = sigma_list[controller.pars.simulation_i]
        gss = gss_list[controller.pars.simulation_i]
        fspeed_PCA = controller.metrics['fspeed_PCA']

        # store the parameters and corresponding speed in results array
        para_search_results_PCA[i][0] = sigma
        para_search_results_PCA[i][1] = gss
        para_search_results_PCA[i][2] = fspeed_PCA

        # fspeed PCA
        if fspeed_PCA > optim_para[0]:
            optim_para[0] = fspeed_PCA  # update fspeed
            optim_para[1] = sigma  # update opti sigma
            optim_para[2] = gss  # update opti gss
            optim_para[3] = controller.pars.simulation_i  # update simulation index

    # Print debug info
    d = 1  # debug

    # Plot the results
    # Plot the heat map of the parameter search (using plot2d)
    labels = ['Noise sigma', 'Feedback strength', 'Forward speed (PCA)']

    plt.figure('2D Parameter Search PCA Fspeed', figsize=[10, 10])
    plot_2d(para_search_results_PCA, labels, cmap='nipy_spectral')  # can maybe find nicer cmap='coolwarm' or other
    plt.title('2D Parameter Search PCA Fspeed')
    plt.show()

    d = 1 # debug
if __name__ == '__main__':
    exercise8()
    plt.show()
