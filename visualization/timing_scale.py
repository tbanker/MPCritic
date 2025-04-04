import os
import numpy as np
import torch

if __name__ == '__main__':
    date = '2025-03-21'
    exp_dict = {
        'H':1, 'lim':10., 'set_initial_guess':True, 'mu_class':'MLP_bounds', 'n_hidden':2, 'hidden_nodes':100
    }
    n_list = [2**i for i in range(2,8)]
    m_list = n_list
    seeds = np.arange(10).astype(np.int64)

    # batch_limit = int(4*1e4+2)
    res_dicts = {f"n={n}_m={m}":{
        "mu_fwd" : [],
        "mu_bkwd" : [],
        "mu_setup": [],
        "dpcontrol_setup": [],
        "critic_setup": [],
        "mpc_setup": [],
        "mpc_init" : [],
        "mpc_fwd" : [],
        "mpc_bkwd" : [],
        "avg_mpc_init" : [],
        "avg_mpc_fwd" : [],
        "avg_mpc_bkwd" : [],
    } for n,m in zip(n_list, m_list)}

    for n,m in zip(n_list, m_list):
        exp_name = f"n={n}_m={m}"
        for seed in seeds:
            dir_name = f"{date}_n={n}_m={m}_H={exp_dict['H']}_lim={exp_dict['lim']}_set_initial_guess={exp_dict['set_initial_guess']}"
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "scalability", dir_name)
            file_path = os.path.join(save_dir, f'seed={seed}.pt')

            results = torch.load(file_path)

            res_dicts[exp_name]['mu_fwd'].append(results['mu_fwd'])
            res_dicts[exp_name]['mu_bkwd'].append(results['mu_bkwd'])
            res_dicts[exp_name]['mu_setup'].append(results['mu_setup'])
            res_dicts[exp_name]['dpcontrol_setup'].append(results['dpcontrol_setup'])
            res_dicts[exp_name]['critic_setup'].append(results['critic_setup'])
            res_dicts[exp_name]['mpc_setup'].append(results['mpc_setup'])
            res_dicts[exp_name]['mpc_init'].append(results['mpc_init'])
            res_dicts[exp_name]['mpc_fwd'].append(results['mpc_fwd'])
            res_dicts[exp_name]['mpc_bkwd'].append(results['mpc_bkwd'])
            res_dicts[exp_name]['avg_mpc_init'].append(results['avg_mpc_init'])
            res_dicts[exp_name]['avg_mpc_fwd'].append(results['avg_mpc_fwd'])
            res_dicts[exp_name]['avg_mpc_bkwd'].append(results['avg_mpc_bkwd'])

        for key in ['mu_fwd', 'mu_bkwd', 'mpc_fwd', 'mpc_bkwd']:
            res_dicts[exp_name][key] = np.array(res_dicts[exp_name][key])
            res_dicts[exp_name][f'{key}_avg'] = np.mean(res_dicts[exp_name][key], axis=0)
            res_dicts[exp_name][f'{key}_std'] = np.std(res_dicts[exp_name][key], axis=0, ddof=1)

    def scientific_format(string):
        if string[-3] == '-':
            return string[:3] + "\\cdot10^{" + string[-3] + string[-1] + "}"
        else:
            return string[:3] + "\\cdot10^{" + string[-1] + "}"

    scale = 1
    exp_keys = list(res_dicts.keys())
    for i, exp_key in enumerate(exp_keys):
        mu_fwd_str = f"{res_dicts[exp_key]['mu_fwd_avg'][-1]/scale:.1e}"
        mu_bkwd_str = f"{res_dicts[exp_key]['mu_bkwd_avg'][-1]/scale:.1e}"
        mpc_fwd_str = f"{res_dicts[exp_key]['mpc_fwd_avg'][-1]/scale:.1e}"
        mpc_bkwd_str = f"{res_dicts[exp_key]['mpc_bkwd_avg'][-1]/scale:.1e}"
        mu_fwd_str, mu_bkwd_str, mpc_fwd_str, mpc_bkwd_str = [scientific_format(string) for string in [mu_fwd_str, mu_bkwd_str, mpc_fwd_str, mpc_bkwd_str]]
        print(f"${exp_key.split('=')[-1]}$ & ${mu_fwd_str}$ & ${mpc_fwd_str}$ & ${mu_bkwd_str}$ & ${mpc_bkwd_str}$\\\\")