from casadi import *
import do_mpc
from do_mpc.data import save_results, load_results
import matplotlib.pyplot as plt

def episode_plot(sim_data, path=''):
    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=3)
    params = {
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

    # Initialize graphic:
    graphics = do_mpc.graphics.Graphics(sim_data)
    graphics.clear()

    fig, ax = plt.subplots(4, sharex=True, layout='constrained')
    # Configure plot:
    graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
    graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
    graphics.add_line(var_type='_aux', var_name='goal', axis=ax[0], linestyle='dotted', color='grey')
    graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
    graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
    graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[2])
    graphics.add_line(var_type='_u', var_name='F', axis=ax[3])
    ax[0].set_ylabel('c [mol/l]')
    ax[1].set_ylabel('T [K]')
#     ax[2].set_ylabel(r'$\Delta$ T [K]')
    ax[2].set_ylabel(r'$\dot{Q}$ [mW]')
    ax[3].set_ylabel('Flow [dal/h]')
    ax[3].set_xlabel('Time [h]')

    label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
    ax[0].legend(label_lines, [r'$C_A$', r'$C_B$'], loc='upper right', ncol=2)
    label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
    ax[1].legend(label_lines, [r'$T_R$', r'$T_K$'], loc='upper right', ncol=2)

    fig.align_ylabels()

    graphics.plot_results()
    graphics.reset_axes()

    

    return fig