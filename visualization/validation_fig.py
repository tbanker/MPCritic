import os
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.style.use('visualization/ieee.mplstyle')
plt.style.use('tableau-colorblind10')
plt.rcParams['text.usetex'] = True

prim_ln_kwargs = {
    'ls' : '-',
    'lw' : 2.0,
    'marker' : 'o',
    'ms' : 5.0,
    'mec' : 'none',
    'mew' : 0.5,
    'mfc' : 'none',
}

sec_ln_kwargs = {
    'ls' : '--',
    'lw' : 2.0,
    'marker' : 'o',
    'ms' : 5.0,
    'mec' : 'none',
    'mew' : 0.5,
    'mfc' : 'none',
}

def plot_loss(ax, x, y, yerr, color, leglabels=None, ln_kwargs=prim_ln_kwargs):
    
    ax.plot(x, y, color=color, label=leglabels, **ln_kwargs)
    ax.fill_between(x, y1=(y+yerr), y2=(y-yerr), color=color, alpha=0.3, lw=0.)

    return ax

def plot_losses(plotdata0, plotdata1a, plotdata1b, ylabels, xlabel):

    x0s, y0s, yerr0s, leglabel0s, color0s = plotdata0.values()
    x1as, y1as, yerr1as, leglabel1as, color1as = plotdata1a.values()
    x1bs, y1bs, yerr1bs, leglabel1bs, color1bs = plotdata1b.values()

    fig, axes = plt.subplots(2, 1, sharex=True, layout='constrained')
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0,1))

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x0s, y0s, yerr0s, leglabel0s, color0s)):
        plot_loss(axes[0], x, y, yerr, color, leglabel, ln_kwargs=prim_ln_kwargs)

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x1as, y1as, yerr1as, leglabel1as, color1as)):
        plot_loss(axes[1], x, y, yerr, color, leglabel, ln_kwargs=prim_ln_kwargs)

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x1bs, y1bs, yerr1bs, leglabel1bs, color1bs)):
        plot_loss(axes[1], x, y, yerr, color, leglabel, ln_kwargs=sec_ln_kwargs)

    axes[0].set_ylabel(ylabels[0])
    axes[0].set_ylim(bottom=0.)
    # axes[0].set_yscale('log')

    axes[1].set_ylabel(ylabels[1])
    axes[1].set_ylim(bottom=0.)
    # axes[1].set_yscale('log')

    axes[1].set_xlabel(xlabel)
    axes[1].set_xlim(left=x[0], right=x[-1])

    axes[0].legend(loc='upper right')
    axes[1].legend(ncols=1, loc='upper right')

    return fig, axes

if __name__ == '__main__':
    date = '2025-03-04'
    exp_dicts = {
        'learn-f' :        {'learn_dynamics':True,  'learn_dpcontrol':False, 'learn_mpcritic':False, 'pd_V':False, },
        'learn-K' :        {'learn_dynamics':False, 'learn_dpcontrol':True,  'learn_mpcritic':False, 'pd_V':False, },
        'learn-P' :        {'learn_dynamics':False, 'learn_dpcontrol':False, 'learn_mpcritic':True,  'pd_V':False, },
        'learn-pd-P' :     {'learn_dynamics':False, 'learn_dpcontrol':False, 'learn_mpcritic':True,  'pd_V':True, },
        'learn-f_K_P' :    {'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':False, },
        'learn-pd-f_K_P' : {'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':True, },
    }
    seeds = np.arange(65).astype(np.int64)
    # mask = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #                  0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #                  0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]) # final MSE for P > 1.
    # seeds = seeds[seeds*mask != 0]
    batch_limit = int(4*1e4+2)

    res_dicts = {key:{
        'f_mses':[],
        'K_mses':[],
        'P_mses':[],
    } for key in exp_dicts}

    for exp_name, exp_dict in exp_dicts.items():
        final_As = []
        final_Bs = []
        final_Ps = []
        final_Ks = []
        for seed in seeds:
            dir_name = f"{date}_f={exp_dict['learn_dynamics']}_mu={exp_dict['learn_dpcontrol']}_V={exp_dict['learn_mpcritic']}_PD={exp_dict['pd_V']}"
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "validation", dir_name)
            file_path = os.path.join(save_dir, f'seed={seed}.pt')

            results = torch.load(file_path)

            A_lrn, B_lrn, K_lrn, P_lrn = results['A_lrn'], results['B_lrn'], results['K_lrn'], results['P_lrn'],
            A_err, B_err, K_err, P_err = A_lrn-results['A_env'], B_lrn-results['B_env'], K_lrn-results['K_opt'], P_lrn-results["P_opt"]
            f_err = np.concatenate([A_err, B_err], axis=-1)
            f_err, K_err, P_err = [M.reshape(*M.shape[:-2], -1) for M in [f_err, K_err, P_err]]
            f_mse, K_mse, P_mse = np.mean(f_err**2, axis=-1), np.mean(K_err**2, axis=-1), np.mean(P_err**2, axis=-1)
            f_mse, K_mse, P_mse = f_mse[:batch_limit], K_mse[:batch_limit], P_mse[:batch_limit]

            res_dicts[exp_name]['f_mses'].append(f_mse)
            res_dicts[exp_name]['K_mses'].append(K_mse)
            res_dicts[exp_name]['P_mses'].append(P_mse)

            final_As.append(A_lrn[-1])
            final_Bs.append(B_lrn[-1])
            final_Ps.append(P_lrn[-1])
            final_Ks.append(K_lrn[-1])

        for key in ['f_mses', 'K_mses', 'P_mses']:
            res_dicts[exp_name][key] = np.array(res_dicts[exp_name][key])
            res_dicts[exp_name][f'{key}_avg'] = np.mean(res_dicts[exp_name][key], axis=0)
            res_dicts[exp_name][f'{key}_std'] = np.std(res_dicts[exp_name][key], axis=0, ddof=1)

    exp_keys = list(exp_dicts.keys())[:4]
    y0s = [res_dicts[key][f"{key.split('-')[-1]}_mses_avg"] for key in exp_keys]
    yerr0s = [2*res_dicts[key][f"{key.split('-')[-1]}_mses_std"] for key in exp_keys]
    x0s = [np.arange(batch_limit)] * len(y0s)
    leglabel0s = [r'$A,B$', r'$K$', r'$P$', r'$P=L^{\top}L + \epsilon I$']
    color0s = ['C0', 'C1', 'C2', 'C5']
    plotdata0 = {'xs':x0s, 'ys':y0s, 'yerrs':yerr0s, 'leglabels':leglabel0s, 'colors':color0s}

    exp_key = 'learn-f_K_P'
    y1as = [res_dicts[exp_key][key] for key in ['f_mses_avg', 'K_mses_avg', 'P_mses_avg']]
    yerr1as = [2*res_dicts[exp_key][key] for key in ['f_mses_std', 'K_mses_std', 'P_mses_std']]
    x1as = [np.arange(batch_limit)] * len(y1as)
    leglabel1as = [None, None, r'$P$']
    color1as = color0s[:3]
    plotdata1a = {'xs':x1as, 'ys':y1as, 'yerrs':yerr1as, 'leglabels':leglabel1as, 'colors':color1as}

    exp_key = 'learn-pd-f_K_P'
    y1bs = [res_dicts[exp_key][key] for key in ['f_mses_avg', 'K_mses_avg', 'P_mses_avg']]
    yerr1bs = [2*res_dicts[exp_key][key] for key in ['f_mses_std', 'K_mses_std', 'P_mses_std']]
    x1bs = [np.arange(batch_limit)] * len(y1bs) # [np.arange(A_lrn.shape[1])] * len(y1bs)
    leglabel1bs = [None, None, r'$P=L^{\top}L + \epsilon I$']
    color1bs = color0s[:2] + color0s[2+1:]
    plotdata1b = {'xs':x1bs, 'ys':y1bs, 'yerrs':yerr1bs, 'leglabels':leglabel1bs, 'colors':color1bs}

    fig, axes = plot_losses(plotdata0, plotdata1a, plotdata1b, ylabels=["Individual MSE", "Co-learning MSE"], xlabel='Batch')

    save_results = True
    save_dir = os.path.join(os.path.dirname(__file__), "validation")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{date}_validation"))