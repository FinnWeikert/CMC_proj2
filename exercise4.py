

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
from util.rw import load_object
import matplotlib.pyplot as plt


def exercise4():

    pylog.info("Ex 4")
    pylog.info("Implement exercise 4")
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 30  # Number of samples
    #base = 2  # Logarithmic base

    # steepnesses = np.logspace(np.log2(0.1), np.log2(100), nsim, base=base)
    I_list = np.linspace(0, 30, nsim)

    pylog.info(
    "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=7501,
            I=I,
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
    log_controlers = os.listdir("logs/exercise4/")
    # Count the number of file
    num_files = len(log_controlers)
    
    # if not corresponding number of simulations stored in logs, run the simulations
    if num_files != nsim:
        controllers = run_multiple(pars_list, num_process=8)
    else: # load the simulations from logs
        controllers = []
        for i in range(nsim):
            controllers.append(load_object("logs/exercise4/controller"+str(i)))

    ptcc_list = []
    for controller in controllers:
        ptcc_list.append(controller.metrics['ptcc'])

    fig1 = plt.figure('ptcc', figsize=(10, 6))
    plt.axhline(y=1.5, color='r', linestyle='--', label='stable oscillation limit')  # Add horizontal line at y=0.033
    plt.axvline(x=1, color='grey', linestyle='-')
    plt.axvline(x=26, color='grey', linestyle='-')
    plt.plot(I_list, ptcc_list, linewidth=3)
    plt.xlabel('input I')
    plt.ylabel('ptcc')
    plt.title('Peak to Peak Correlation as Function of I')
    plt.legend(fontsize=10)  # Adjust legend size
    plt.grid(True)
    # Add shaded region between the two vertical lines
    plt.fill_betweenx(np.linspace(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], 100), 1, 26, color='lightgrey')
    # Add text annotations for x-values at the level of the x-axis
    plt.text(1, plt.gca().get_ylim()[0]-0.02, 'I=1', ha='left', va='top', color='grey')
    plt.text(26, plt.gca().get_ylim()[0]-0.02, 'I=26', ha='left', va='top', color='grey')
    # save_figure(fig1, dir='results/ex4', extensions=['png'])
    #plt.show()
    

    # HOW does the frequency and wave frequency change in range I = [1, 26] ?
    I_list = np.linspace(1,25,25)

    freq_list = []
    wavefreq_list = []
    for i, controller in enumerate(controllers):
        if i >= I_list[0] and i <= I_list[-1]:
            freq_list.append(controller.metrics['frequency'])
            wavefreq_list.append(controller.metrics['wavefrequency'])

    fig2 = plt.figure('Frequencies',figsize=(10, 6))
    plt.plot(I_list, freq_list, label='frequency', linewidth=2.5)
    plt.plot(I_list, wavefreq_list, label='wavefrequency', linewidth=2.5)
    plt.xlabel('input I')
    plt.ylabel('[Hz]')
    plt.title('Frequency and Wavefrequency as Function of I')
    plt.legend(fontsize=10)  # Adjust legend size
    plt.grid(True)
    # save_figure(fig2, dir='results/ex4')
    # plt.show()


    # FINIS, sauf save_figure marche pas!

if __name__ == '__main__':
    exercise4()
    plt.show()

