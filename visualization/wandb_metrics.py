import wandb
import pandas as pd

wandb.login()
api = wandb.Api()
entity= "mpcritic-final"
runs = api.runs(entity)

config_list, name_list, df_list = [], [], []
for run in runs:
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    # extract and append relevant run data
    return_history = run.scan_history(keys=['charts/episodic_return'])
    return_list = [row['charts/episodic_return'] for row in return_history]
    toc_history = run.scan_history(keys=['charts/time_outside_constraints'])
    toc_list = [row['charts/time_outside_constraints'] for row in toc_history]
    toc_list = toc_list[:-1] if len(toc_list) > len(return_list) else toc_list
    print(len(return_list), len(toc_list))
    df_list.append(pd.DataFrame.from_dict({
        'charts/episodic_return' : return_list,
        'charts/time_outside_constraints' : toc_list
    }, orient='columns'))

mode_name_map = {}
for name, config in zip(name_list, config_list):
    mode_name_map.setdefault(config['critic_mode'], []).append(name)

name_data_map = {name_list[i]:df_list[i] for i in range(len(runs))}

mpc_data_df = pd.concat([name_data_map[name] for name in mode_name_map['mpcritic']], axis=1, keys=mode_name_map['mpcritic'])
rl_data_df = pd.concat([name_data_map[name] for name in mode_name_map['vanilla']], axis=1, keys=mode_name_map['vanilla'])


n = 10
res_df = pd.DataFrame(columns=['name','critic_mode','avg_n_returns','max_n_toc','min_n_toc'])
for mode, names in mode_name_map.items():
    for name in names:
        returns_mask = ~name_data_map[name]['charts/episodic_return'].isnull()
        toc_mask = ~name_data_map[name]['charts/time_outside_constraints'].isnull()

        returns = name_data_map[name]['charts/episodic_return'][returns_mask]
        toc = name_data_map[name]['charts/time_outside_constraints'][toc_mask]

        res = pd.DataFrame.from_dict({
            'name' : [name],
            'critic_mode' : [mode],
            'avg_n_returns' : [returns[-n:].mean()],
            'max_n_toc' : [toc[-n:].max()],
            'min_n_toc' : [toc[-n:].max()],
            # 'name' : [name]*n,
            # 'critic_mode' : [mode]*n,
            # 'returns' : returns[-n:].values,
            # 'toc' : toc[-n:].values,
        }, orient='columns')

        res_df = pd.concat([res_df, res], ignore_index=True)

rl_df = res_df[res_df['critic_mode'] == 'vanilla']
print(f"rl: {rl_df['avg_n_returns'].mean()} {rl_df['avg_n_returns'].std()} {rl_df['min_n_toc'].min()} {rl_df['max_n_toc'].max()}")
mpc_df = res_df[res_df['critic_mode'] == 'mpcritic']
print(f"mpc: {mpc_df['avg_n_returns'].mean()} {mpc_df['avg_n_returns'].std()} {mpc_df['min_n_toc'].min()} {mpc_df['max_n_toc'].max()}")

rl_avg_return = res_df[res_df['critic_mode'] == 'vanilla']['avg_n_returns'].mean()

series = []
for i, name in enumerate(mode_name_map['mpcritic']):
    returns_mask = ~name_data_map[name]['charts/episodic_return'].isnull()
    returns = name_data_map[name]['charts/episodic_return'][returns_mask]
    toc_mask = ~name_data_map[name]['charts/time_outside_constraints'].isnull()
    toc = name_data_map[name]['charts/time_outside_constraints']
    # starting from terminal episode, count backwards
    returns = returns.reset_index(drop=True)
    rolling_returns = returns.rolling(n).mean()[::-1].reset_index(drop=True)
    rolling_returns.name = name
    series.append(rolling_returns.copy())

rolling_df = pd.concat(series, axis=1)
mpc_idx = (rolling_df.mean(axis=1) > rl_avg_return).idxmin()

epi_length = 50
learning_starts = 5e3
print(f"mpc>=rl {mpc_idx} episodes ({int(mpc_idx*epi_length)} steps) before rl_max (learning starts at {int(learning_starts)})")
