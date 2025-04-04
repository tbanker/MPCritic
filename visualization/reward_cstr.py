import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use('visualization/ieee.mplstyle')
plt.style.use('tableau-colorblind10')


def df_stats(df, window=10):

    df_stats = pd.DataFrame()

    df_stats['mean'] = df.mean(axis=1).rolling(window).mean()
    df_stats['sem'] = df.sem(axis=1).rolling(window).mean()
    df_stats['std'] = df.std(axis=1).rolling(window).mean()
    df_stats['median'] = df.median(axis=1).rolling(window).mean()
    df_stats['quantile_upper'] = df.quantile(q=0.75, axis=1).rolling(window).mean()
    df_stats['quantile_lower'] = df.quantile(q=0.25, axis=1).rolling(window).mean()

    # print(df_stats.head())
    return df_stats

def clean_wandb_csv(path):
    df = pd.read_csv(path)
    df = df.iloc[:, ~df.columns.str.contains('__MIN')]
    df = df.iloc[:, ~df.columns.str.contains('__MAX')]
    df = df.drop('Step', axis=1) 
    return pd.DataFrame(df.values)

if __name__ == '__main__':

    data_mpcritic = 'data/mpcritic_cstr_sac_unconstrained_v2.csv'
    data_vanilla = 'data/vanilla_cstr_sac_unconstrained.csv'
    # df_mpcritic = pd.read_csv(data_mpcritic)
    # df_mpcritic = df_mpcritic.iloc[:, ~df_mpcritic.columns.str.contains('__MIN')]
    df_mpcritic = clean_wandb_csv(data_mpcritic)
    df_vanilla = clean_wandb_csv(data_vanilla)


    df_mpcritic_stats = df_stats(df_mpcritic, window=20)
    df_vanilla_stats = df_stats(df_vanilla, window=20)

    # fig, axes = plt.subplots(1, 1, sharex=True, )
    fig, ax = plt.subplots(layout='constrained')
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,1))
    x = range(len(df_mpcritic_stats['median']))
    ax.set_xlim(x[0]+1,x[-1]+1)
    locs, labels = plt.xticks()
    # plt.xticks(ticks=locs, labels=locs*50)
    # print(locs)
    # ax.set_xticklabels()
    plt.plot(x, df_mpcritic_stats['median'], label='SAC+'+r'\texttt{MPCritic}')
    plt.fill_between(x, y1=df_mpcritic_stats['quantile_lower'], y2=df_mpcritic_stats['quantile_upper'], alpha=0.3, lw=1.)
    plt.plot(x, df_vanilla_stats['median'], label='SAC')
    plt.fill_between(x, y1=df_vanilla_stats['quantile_lower'], y2=df_vanilla_stats['quantile_upper'], alpha=0.3, lw=1.)
    plt.legend(ncol=2, loc='upper center')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.savefig('data/cstr_reward')
    plt.savefig('data/cstr_reward.pdf')


