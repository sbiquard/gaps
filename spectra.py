#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update(
    {'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.grid': 'true'}
)

RUNS = Path('./gaps-scripts/runs')

# make plots directory if it does not exist
Path('plots').mkdir(exist_ok=True)


def get_noise_cl(run: Path):
    cl = np.load(run / 'spectra/noise_cl.npz')
    return {'ell': cl['ell_arr'], 'TT': cl['cl_00'][0], 'EE': cl['cl_22'][0], 'BB': cl['cl_22'][3]}


# HWP reference runs
hwp_ins_ref = RUNS / 'hwp/ins/reference'
hwp_atm_ref = RUNS / 'hwp/atm/reference'
cl_ins_ref = get_noise_cl(hwp_ins_ref)
cl_atm_ref = get_noise_cl(hwp_atm_ref)

ell = cl_ins_ref['ell']
ell_range = (ell > 30) & (ell < 1_000)

# NoHWP reference runs
nohwp_ins_ref = RUNS / 'nohwp/ins/reference'
nohwp_atm_ref = RUNS / 'nohwp/atm/reference'
cl_ins_nohwp_ref = get_noise_cl(nohwp_ins_ref)
cl_atm_nohwp_ref = get_noise_cl(nohwp_atm_ref)


# Create a 2x3 subplot grid to show both HWP and NoHWP configurations
with sns.color_palette('Set2'):
    fig, axs = plt.subplots(2, 3, sharex=True, layout='constrained', figsize=(10, 6))

    # Top row: HWP configuration
    for i, spec in enumerate(['TT', 'EE', 'BB']):
        if i == 0:
            # Use semilogy for TT in hwp case
            axs[0, i].semilogy(ell[ell_range], cl_ins_ref[spec][ell_range], label='Instrumental')
            axs[0, i].semilogy(ell[ell_range], cl_atm_ref[spec][ell_range], label='Atmospheric')
        else:
            # Use regular plot for EE and BB in hwp case
            axs[0, i].plot(ell[ell_range], cl_ins_ref[spec][ell_range])
            axs[0, i].plot(ell[ell_range], cl_atm_ref[spec][ell_range])

    # Bottom row: NoHWP configuration
    for i, spec in enumerate(['TT', 'EE', 'BB']):
        # Use semilogy for all plots in nohwp case
        axs[1, i].semilogy(ell[ell_range], cl_ins_nohwp_ref[spec][ell_range], label='Instrumental')
        axs[1, i].semilogy(ell[ell_range], cl_atm_nohwp_ref[spec][ell_range], label='Atmospheric')

    # Set titles and labels
    for i, _title in enumerate(['TT', 'EE', 'BB']):
        axs[0, i].set_title(_title)
        # Only show x-label on second row
        axs[1, i].set_xlabel(r'Multipole $\ell$')

    # Set y-axis labels
    axs[0, 0].set_ylabel('HWP\n$N_\\ell [\\mu K^2]$')
    axs[1, 0].set_ylabel('No HWP\n$N_\\ell [\\mu K^2]$')

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')

    # Save figure
    fig.savefig('plots/reference_runs_comparison.svg')

HWP_INS = RUNS / 'hwp/ins'
runs_hwp_ins = {
    'ref': HWP_INS / 'reference',
    'cond': HWP_INS / 'gapfill-cond',
    'offset': HWP_INS / 'gapfill-marg',
    'offset-real': HWP_INS / 'gapfill-marg-real-noise',
    'offset-noise-only': HWP_INS / 'gapfill-marg-noise-only',
    'mirror': HWP_INS / 'gapfill-mirror',
    'nested': HWP_INS / 'nested',
}
cl_hwp_ins = {k: get_noise_cl(_run) for k, _run in runs_hwp_ins.items()}

HWP_ATM = RUNS / 'hwp/atm'
runs_hwp_atm = {
    'ref': HWP_ATM / 'reference',
    'cond': HWP_ATM / 'gapfill-cond',
    'offset': HWP_ATM / 'gapfill-marg',
    'offset-real': HWP_ATM / 'gapfill-marg-real-noise',
    'offset-noise-only': HWP_ATM / 'gapfill-marg-noise-only',
    'mirror': HWP_ATM / 'gapfill-mirror',
    # 'nested': HWP_ATM / 'nested',
}
cl_hwp_atm = {k: get_noise_cl(_run) for k, _run in runs_hwp_atm.items()}

NOHWP_INS = RUNS / 'nohwp/ins'
runs_nohwp_ins = {
    'ref': NOHWP_INS / 'reference',
    'cond': NOHWP_INS / 'gapfill-cond',
    'offset': NOHWP_INS / 'gapfill-marg',
    'offset-real': NOHWP_INS / 'gapfill-marg-real-noise',
    'offset-noise-only': NOHWP_INS / 'gapfill-marg-noise-only',
    'mirror': NOHWP_INS / 'gapfill-mirror',
    'nested': NOHWP_INS / 'nested',
}
cl_nohwp_ins = {k: get_noise_cl(_run) for k, _run in runs_nohwp_ins.items()}

NOHWP_ATM = RUNS / 'nohwp/atm'
runs_nohwp_atm = {
    'ref': NOHWP_ATM / 'reference',
    'cond': NOHWP_ATM / 'gapfill-cond',
    'offset': NOHWP_ATM / 'gapfill-marg',
    'offset-real': NOHWP_ATM / 'gapfill-marg-real-noise',
    'offset-noise-only': NOHWP_ATM / 'gapfill-marg-noise-only',
    'mirror': NOHWP_ATM / 'gapfill-mirror',
    # 'nested': NOHWP_ATM / 'nested',
}
cl_nohwp_atm = {k: get_noise_cl(_run) for k, _run in runs_nohwp_atm.items()}


def plot_comparison(cl_dict_hwp, cl_dict_nohwp, ref_key, output_file):
    fig, axs = plt.subplots(2, 3, sharex=True, layout='constrained', figsize=(10, 6))
    cm = sns.color_palette(as_cmap=True)

    # Top row: HWP configuration
    for i, tt_ee_bb in enumerate(['TT', 'EE', 'BB']):
        ax = axs[0, i]
        ax.set(title=tt_ee_bb)
        ax.axhline(y=1.0, c='k', ls='--')
        for ik, k in enumerate(['cond', 'offset-real', 'offset-noise-only', 'mirror', 'nested']):
            if k not in cl_dict_hwp:
                continue
            ax.plot(
                ell[ell_range],
                cl_dict_hwp[k][tt_ee_bb][ell_range] / cl_dict_hwp[ref_key][tt_ee_bb][ell_range],
                color=cm[ik],
                label=k,
            )
        ax.set(yscale='log')
        if i == 0:
            ax.set(ylabel='HWP\nRatio')

    # Bottom row: NoHWP configuration
    for i, tt_ee_bb in enumerate(['TT', 'EE', 'BB']):
        ax = axs[1, i]
        ax.axhline(y=1.0, c='k', ls='--')
        ax.set(xlabel=r'Multipole $\ell$')
        for ik, k in enumerate(['cond', 'offset-real', 'offset-noise-only', 'mirror', 'nested']):
            if k not in cl_dict_nohwp:
                continue
            ax.plot(
                ell[ell_range],
                cl_dict_nohwp[k][tt_ee_bb][ell_range] / cl_dict_nohwp[ref_key][tt_ee_bb][ell_range],
                color=cm[ik],
                label=k,
            )
        ax.set(yscale='log')
        if i == 0:
            ax.set(ylabel='No HWP\nRatio')

    # Add title and legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')
    fig.savefig(output_file)


# Combined plots for instrumental noise (HWP and NoHWP)
plot_comparison(cl_hwp_ins, cl_nohwp_ins, 'ref', 'plots/instrumental_comparison.svg')

# Combined plots for atmospheric noise (HWP and NoHWP)
plot_comparison(cl_hwp_atm, cl_nohwp_atm, 'ref', 'plots/atmospheric_comparison.svg')


def plot_gap_filling_comparison_ratio(cl_dicts, methods, output_file):
    """Generate plots comparing different gap filling methods as a ratio over the noise-only case.

    Args:
        cl_dicts: Dictionary with 'hwp' and 'nohwp' keys, each containing CL data
        methods: Dictionary mapping method keys to display labels
    """
    fig, axs = plt.subplots(2, 3, sharex=True, layout='constrained', figsize=(10, 6))

    # Process HWP (top row) and NoHWP (bottom row)
    for row, config in enumerate(['hwp', 'nohwp']):
        for col, tt_ee_bb in enumerate(['TT', 'EE', 'BB']):
            ax = axs[row, col]
            ax.set(title=tt_ee_bb if row == 0 else '')
            if row == 1:  # Only add x-label to bottom row
                ax.set(xlabel=r'Multipole $\ell$')

            cl_dict = cl_dicts[config]

            # Add horizontal reference line at y=1.0
            ax.axhline(y=1.0, c='k', ls='--', alpha=0.5)

            # Get the noise-only data for normalization
            noise_only_key = 'offset-noise-only'
            if noise_only_key in cl_dict:
                noise_only_data = cl_dict[noise_only_key][tt_ee_bb][ell_range]

                # Plot each method as ratio over noise-only
                for method, label in methods.items():
                    if method in cl_dict and method != noise_only_key:
                        ax.plot(
                            ell[ell_range],
                            cl_dict[method][tt_ee_bb][ell_range] / noise_only_data,
                            label=label,
                        )

    # Set y-axis labels for first column
    axs[0, 0].set(ylabel='HWP\nRatio to noise-only', yscale='log')
    axs[1, 0].set(ylabel='No HWP\nRatio to noise-only', yscale='log')

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')

    fig.savefig(output_file)


# Define gap filling methods with their labels
gap_filling_methods = {
    'offset': 'original',
    'offset-real': 'realistic',
    'offset-noise-only': 'noise only',
}

# Plot for instrumental noise (both HWP and NoHWP)
plot_gap_filling_comparison_ratio(
    {'hwp': cl_hwp_ins, 'nohwp': cl_nohwp_ins},
    gap_filling_methods,
    'plots/instrumental_gap_filling_ratio_comparison.svg',
)

# Plot for atmospheric noise (both HWP and NoHWP)
plot_gap_filling_comparison_ratio(
    {'hwp': cl_hwp_atm, 'nohwp': cl_nohwp_atm},
    gap_filling_methods,
    'plots/atmospheric_gap_filling_ratio_comparison.svg',
)
