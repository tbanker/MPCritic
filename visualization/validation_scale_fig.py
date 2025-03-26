import os
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.style.use('visualization/ieee.mplstyle')
plt.style.use('tableau-colorblind10')

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

def plot_losses(plotdata0, plotdata1b, ylabels, xlabel):

    x0s, y0s, yerr0s, leglabel0s, color0s = plotdata0.values()
    # x1as, y1as, yerr1as, leglabel1as, color1as = plotdata1a.values()
    x1bs, y1bs, yerr1bs, leglabel1bs, color1bs = plotdata1b.values()

    fig, axes = plt.subplots(2, 1, sharex=True, layout='constrained')
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0,1))

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x0s, y0s, yerr0s, leglabel0s, color0s)):
        plot_loss(axes[0], x, y, yerr, color, leglabel, ln_kwargs=prim_ln_kwargs)

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x1bs, y1bs, yerr1bs, leglabel1bs, color1bs)):
        plot_loss(axes[1], x, y, yerr, color, leglabel, ln_kwargs=prim_ln_kwargs)

    axes[0].set_ylabel(ylabels[0])
    axes[0].set_ylim(bottom=0.)
    # axes[0].set_yscale('log')

    axes[1].set_ylabel(ylabels[1])
    axes[1].set_ylim(bottom=0.)
    # axes[1].set_yscale('log')

    axes[1].set_xlabel(xlabel)
    # axes[1].set_xlim(left=x[0], right=x[-1])
    axes[1].set_xscale('log')
    axes[1].set_xlim(right=x[-1])

    # axes[0].legend(loc='upper right', title=r"$n,m$", labelspacing=0.2)
    axes[0].legend(loc='upper right', labelspacing=0.2)
    axes[1].legend(loc='upper right', title=r"$n,m$", labelspacing=0.2)

    fig.supylabel('RMSE', va='baseline')

    return fig, axes

def plot_losses(plotdata0, ylabel, xlabel):

    x0s, y0s, yerr0s, leglabel0s, color0s = plotdata0.values()

    fig, ax = plt.subplots(1, 1, figsize=(3.3,2), sharex=True, layout='constrained')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,1))

    for i, (x, y, yerr, leglabel, color) in enumerate(zip(x0s, y0s, yerr0s, leglabel0s, color0s)):
        plot_loss(ax, x, y, yerr, color, leglabel, ln_kwargs=prim_ln_kwargs)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.)
    # ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    # axes[1].set_xlim(left=x[0], right=x[-1])
    ax.set_xscale('log')
    ax.set_xlim(right=x[-1])

    ax.legend(loc='upper right', title=r"$n,m$", labelspacing=0.4)

    return fig, ax

if __name__ == '__main__':
    date = '2025-03-20'
    exp_dict = {
        'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':True,
    }
    n_list = [2**i for i in range(2,8)] # [4**i for i in range(1,4)] + [4]*2 + [4**i for i in range(2,4)]
    m_list = n_list # 3*[4] + [4**i for i in range(2,4)] + [4**i for i in range(2,4)]
    # n_list = [4]*3 + [4**i for i in range(2,4)]
    # m_list = [4**i for i in range(1,4)] + [4**i for i in range(2,4)]
    seeds = np.arange(20).astype(np.int64)

    # batch_limit = int(4*1e4+2)
    res_dicts = {f"n={n}_m={m}":{
        'f_maes':[],
        'K_maes':[],
        'P_maes':[],
        'cl_maes':[],
        'f_rmses':[],
        'K_rmses':[],
        'P_rmses':[],
        'cl_rmses':[],
        'f_relerrs':[],
        'K_relerrs':[],
        'P_relerrs':[],
        'cl_relerrs':[],
    } for n,m in zip(n_list, m_list)}

    for n,m in zip(n_list, m_list):
        exp_name = f"n={n}_m={m}"
        final_As = []
        final_Bs = []
        final_Ps = []
        final_Ks = []
        for seed in seeds:
            dir_name = f"{date}_n={n}_m={m}_f={exp_dict['learn_dynamics']}_mu={exp_dict['learn_dpcontrol']}_V={exp_dict['learn_mpcritic']}_PD={exp_dict['pd_V']}"
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "validation", dir_name)
            file_path = os.path.join(save_dir, f'seed={seed}.pt')

            results = torch.load(file_path)

            A_lrn, B_lrn, K_lrn, P_lrn = results['A_lrn'], results['B_lrn'], results['K_lrn'], results['P_lrn'],

            A_err, B_err, K_err, P_err = A_lrn-results['A_env'], B_lrn-results['B_env'], K_lrn-results['K_opt'], P_lrn-results["P_opt"]
            f_err = np.concatenate([A_err, B_err], axis=-1)
            f_err, K_err, P_err = [M.reshape(*M.shape[:-2], -1) for M in [f_err, K_err, P_err]]
            f_mae, K_mae, P_mae = np.mean(np.abs(f_err), axis=-1), np.mean(np.abs(K_err), axis=-1), np.mean(np.abs(P_err), axis=-1)
            f_rmse, K_rmse, P_rmse = np.mean(f_err**2, axis=-1)**0.5, np.mean(K_err**2, axis=-1)**0.5, np.mean(P_err**2, axis=-1)**0.5
            # f_rmse, K_rmse, P_rmse = f_rmse[:batch_limit], K_rmse[:batch_limit], P_rmse[:batch_limit]

            cl_lrn = A_lrn + B_lrn @ K_lrn
            cl_opt = results['A_env'] + results['B_env'] @ results['K_opt']
            cl_err = (cl_lrn - cl_opt).reshape(*cl_lrn.shape[:-2],-1)
            cl_mae = np.mean(np.abs(cl_err), axis=-1)
            cl_rmse = np.mean(cl_err**2, axis=-1)**0.5

            f_frob = np.linalg.norm(np.concatenate([results['A_env'],results['B_env']], axis=-1), ord='fro')
            K_frob = np.linalg.norm(results['K_opt'], ord='fro')
            P_frob = np.linalg.norm(results['P_opt'], ord='fro')
            cl_frob = np.linalg.norm(cl_opt, ord='fro')
            f_relerr, K_relerr, P_relerr, cl_relerr = (f_rmse / f_frob), (K_rmse / K_frob), (P_rmse / P_frob), (cl_rmse / cl_frob)

            res_dicts[exp_name]['f_maes'].append(f_mae)
            res_dicts[exp_name]['K_maes'].append(K_mae)
            res_dicts[exp_name]['P_maes'].append(P_mae)
            res_dicts[exp_name]['cl_maes'].append(cl_mae)
            res_dicts[exp_name]['f_rmses'].append(f_rmse)
            res_dicts[exp_name]['K_rmses'].append(K_rmse)
            res_dicts[exp_name]['P_rmses'].append(P_rmse)
            res_dicts[exp_name]['cl_rmses'].append(cl_rmse)
            res_dicts[exp_name]['f_relerrs'].append(f_relerr)
            res_dicts[exp_name]['K_relerrs'].append(K_relerr)
            res_dicts[exp_name]['P_relerrs'].append(P_relerr)
            res_dicts[exp_name]['cl_relerrs'].append(cl_relerr)

            final_As.append(A_lrn[-1])
            final_Bs.append(B_lrn[-1])
            final_Ps.append(P_lrn[-1])
            final_Ks.append(K_lrn[-1])

        for key in ['f_maes', 'K_maes', 'P_maes', 'cl_maes', 'f_rmses', 'K_rmses', 'P_rmses', 'cl_rmses', 'f_relerrs', 'K_relerrs', 'P_relerrs', 'cl_relerrs']:
            res_dicts[exp_name][key] = np.array(res_dicts[exp_name][key])
            res_dicts[exp_name][f'{key}_avg'] = np.mean(res_dicts[exp_name][key], axis=0)
            res_dicts[exp_name][f'{key}_std'] = np.std(res_dicts[exp_name][key], axis=0, ddof=1)

    # A-BK closed-loop for all n,m
    # exp_keys = list(res_dicts.keys())
    # y0s = [res_dicts[key]["cl_rmses_avg"] for key in exp_keys]
    # yerr0s = [2*res_dicts[key]["cl_rmses_std"] for key in exp_keys]
    # x0s = [100*np.arange(len(res_dicts[exp_keys[0]]["P_rmses_avg"]))] * len(y0s)
    # leglabel0s = [key.split('=')[-1] for key in exp_keys] # [r"$n=m=$ "+key.split('=')[-1] for key in exp_keys] # [None, None, r'$P=L^{\top}L + \epsilon I$']
    # color0s = [f'C{i}' for i in range(len(exp_keys))]

    # # AB, K, P for a n,m
    exp_key = 'n=128_m=128'
    y0s = [res_dicts[exp_key][key] for key in ['f_rmses_avg', 'K_rmses_avg', 'P_rmses_avg']]
    yerr0s = [2*res_dicts[exp_key][key] for key in ['f_rmses_std', 'K_rmses_std', 'P_rmses_std']]
    # y0s = [res_dicts[exp_key][key] for key in ['f_relerrs_avg', 'K_relerrs_avg', 'P_relerrs_avg']]
    # yerr0s = [2*res_dicts[exp_key][key] for key in ['f_relerrs_std', 'K_relerrs_std', 'P_relerrs_std']]
    # y0s = [res_dicts[exp_key][key] for key in ['f_maes_avg', 'K_maes_avg', 'P_maes_avg']]
    # yerr0s = [2*res_dicts[exp_key][key] for key in ['f_maes_std', 'K_maes_std', 'P_maes_std']]
    x0s = [100*np.arange(len(res_dicts[exp_key]["P_rmses_avg"]))] * len(y0s)
    leglabel0s = [r'$A,B$', r'$K$', r'$P$']
    color0s = [f'C{i}' for i in range(len(leglabel0s))]

    plotdata0 = {'xs':x0s, 'ys':y0s, 'yerrs':yerr0s, 'leglabels':leglabel0s, 'colors':color0s}

    # unused/unplotted
    # exp_key = 'learn-f_K_P'
    # y1as = [res_dicts[exp_key][key] for key in ['f_rmses_avg', 'K_rmses_avg', 'P_rmses_avg']]
    # yerr1as = [2*res_dicts[exp_key][key] for key in ['f_rmses_std', 'K_rmses_std', 'P_rmses_std']]
    # x1as = [np.arange(batch_limit)] * len(y1as)
    # leglabel1as = [None, None, r'$P$']
    # color1as = color0s[:3]
    # plotdata1a = {'xs':x1as, 'ys':y1as, 'yerrs':yerr1as, 'leglabels':leglabel1as, 'colors':color1as}
    plotdata1a = {}

    # P for all n,m
    # y1bs = [res_dicts[key]["P_rmses_avg"] for key in exp_keys]
    # yerr1bs = [2*res_dicts[key]["P_rmses_std"] for key in exp_keys]
    # x1bs = x0s
    # leglabel1bs = [None]*len(exp_keys)
    # color1bs = color0s # color0s[:2] + color0s[2+1:]

    # A-BK closed-loop for all n,m
    exp_keys = list(res_dicts.keys())
    y1bs = [res_dicts[key]["cl_rmses_avg"] for key in exp_keys]
    yerr1bs = [2*res_dicts[key]["cl_rmses_std"] for key in exp_keys]
    # y1bs = [res_dicts[key]["cl_relerrs_avg"] for key in exp_keys]
    # yerr1bs = [2*res_dicts[key]["cl_relerrs_std"] for key in exp_keys]
    # y1bs = [res_dicts[key]["cl_maes_avg"] for key in exp_keys]
    # yerr1bs = [2*res_dicts[key]["cl_maes_std"] for key in exp_keys]
    x1bs = [x0s[0]] * len(y1bs)
    leglabel1bs = [key.split('=')[-1] for key in exp_keys] # [r"$n=m=$ "+key.split('=')[-1] for key in exp_keys] # [None, None, r'$P=L^{\top}L + \epsilon I$']
    color1bs = [f'C{i}' for i in range(len(leglabel0s),len(leglabel0s)+len(exp_keys))]

    plotdata1b = {'xs':x1bs, 'ys':y1bs, 'yerrs':yerr1bs, 'leglabels':leglabel1bs, 'colors':color1bs}

    scale = 1e-5
    for i, exp_key in enumerate(exp_keys):
        # AB_str = f"{res_dicts[exp_key]['f_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['f_rmses_std'][-1]/scale:.1f}"
        # K_str = f"{res_dicts[exp_key]['K_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['K_rmses_std'][-1]/scale:.1f}"
        # P_str = f"{res_dicts[exp_key]['P_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['P_rmses_std'][-1]/scale:.1f}"
        # cl_str = f"{res_dicts[exp_key]['cl_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['cl_rmses_std'][-1]/scale:.1f}"
        # print(f"${exp_key.split('=')[-1]}$ & ${AB_str}$ & ${K_str}$ & ${P_str}$ & ${cl_str}$ \\\\")
        print(f"{exp_key}:\n\
                A,B rmse: {res_dicts[exp_key]['f_rmses_avg'][-1]:.2e}+-{2*res_dicts[exp_key]['f_rmses_std'][-1]:.2e},\n\
                K rmse: {res_dicts[exp_key]['K_rmses_avg'][-1]:.2e}+-{2*res_dicts[exp_key]['K_rmses_std'][-1]:.2e},\n\
                P rmse: {res_dicts[exp_key]['P_rmses_avg'][-1]:.2e}+-{2*res_dicts[exp_key]['P_rmses_std'][-1]:.2e},\n\
                cl rmse: {res_dicts[exp_key]['cl_rmses_avg'][-1]:.2e}+-{2*res_dicts[exp_key]['cl_rmses_std'][-1]:.2e}\n")
                # A,B rmse: {res_dicts[exp_key]['f_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['f_rmses_std'][-1]/scale:.1f},\n\
                # K rmse: {res_dicts[exp_key]['K_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['K_rmses_std'][-1]/scale:.1f},\n\
                # P rmse: {res_dicts[exp_key]['P_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['P_rmses_std'][-1]/scale:.1f},\n\
                # cl rmse: {res_dicts[exp_key]['cl_rmses_avg'][-1]/scale:.1f}\pm{2*res_dicts[exp_key]['cl_rmses_std'][-1]/scale:.1f}\n")

    # fig, axes = plot_losses(plotdata0, plotdata1a, plotdata1b, ylabels=["$M$", "$A-BK$"], xlabel='Batch Number')
    fig, axes = plot_losses(plotdata1b, ylabel="$A-BK$ (RMSE)", xlabel='Batched Learning Steps') # "$||A-BK-(A^{\star}-B^{\star}K^{\star}||_{F})$"

    save_dir = os.path.join(os.path.dirname(__file__), "validation")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{date}_validation_scaling"))