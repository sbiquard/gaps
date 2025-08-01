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
# METHODS = ['reference', 'cond', 'marg', 'mirror', 'nested']

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
    fig, axs = plt.subplots(2, 3, sharex=True, layout='constrained', figsize=(10, 7))

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
    axs[0, 0].set_ylabel(r'$N_\ell [\mu K^2]$')
    axs[1, 0].set_ylabel(r'$N_\ell [\mu K^2]$')

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')

    # Save figure
    fig.savefig('plots/reference_runs_comparison.svg')


HWP_INS = RUNS / 'hwp/ins'
runs_hwp_ins = {
    'ref': HWP_INS / 'reference',
    'cond': HWP_INS / 'gapfill-cond',
    'marg': HWP_INS / 'gapfill-marg',
    'marg-real': HWP_INS / 'gapfill-marg-real-noise',
    'marg-noise-only': HWP_INS / 'gapfill-marg-noise-only',
    'mirror': HWP_INS / 'gapfill-mirror',
    'nested': HWP_INS / 'nested',
}
cl_hwp_ins = {k: get_noise_cl(_run) for k, _run in runs_hwp_ins.items()}

HWP_ATM = RUNS / 'hwp/atm'
runs_hwp_atm = {
    'ref': HWP_ATM / 'reference',
    'cond': HWP_ATM / 'gapfill-cond',
    'marg': HWP_ATM / 'gapfill-marg',
    'marg-real': HWP_ATM / 'gapfill-marg-real-noise',
    # 'marg-noise-only': HWP_ATM / 'gapfill-marg-noise-only',
    'mirror': HWP_ATM / 'gapfill-mirror',
    # 'nested': HWP_ATM / 'nested',
}
cl_hwp_atm = {k: get_noise_cl(_run) for k, _run in runs_hwp_atm.items()}

NOHWP_INS = RUNS / 'nohwp/ins'
runs_nohwp_ins = {
    'ref': NOHWP_INS / 'reference',
    'cond': NOHWP_INS / 'gapfill-cond',
    'marg': NOHWP_INS / 'gapfill-marg',
    'marg-real': NOHWP_INS / 'gapfill-marg-real-noise',
    'marg-noise-only': NOHWP_INS / 'gapfill-marg-noise-only',
    'mirror': NOHWP_INS / 'gapfill-mirror',
    'nested': NOHWP_INS / 'nested',
}
cl_nohwp_ins = {k: get_noise_cl(_run) for k, _run in runs_nohwp_ins.items()}

NOHWP_ATM = RUNS / 'nohwp/atm'
runs_nohwp_atm = {
    'ref': NOHWP_ATM / 'reference',
    'cond': NOHWP_ATM / 'gapfill-cond',
    'marg': NOHWP_ATM / 'gapfill-marg',
    # 'marg-real': NOHWP_ATM / 'gapfill-marg-real-noise',
    # 'marg-noise-only': NOHWP_ATM / 'gapfill-marg-noise-only',
    # 'mirror': NOHWP_ATM / 'gapfill-mirror',
    # 'nested': NOHWP_ATM / 'nested',
}
cl_nohwp_atm = {k: get_noise_cl(_run) for k, _run in runs_nohwp_atm.items()}


def plot_comparison(cl_dict, ref_key, output_file):
    fig, axs = plt.subplots(1, 3, sharex=True, layout='constrained', figsize=(10, 4))
    for i, ax in enumerate(axs):
        tt_ee_bb = ['TT', 'EE', 'BB'][i]
        ax.set(title=tt_ee_bb, xlabel=r'Multipole $\ell$')
        ax.axhline(y=1.0, c='k', ls='--')
        for k, cl in cl_dict.items():
            if k == ref_key:
                continue
            ax.plot(
                ell[ell_range],
                cl[tt_ee_bb][ell_range] / cl_dict[ref_key][tt_ee_bb][ell_range],
                label=k,
            )
    axs[0].set(yscale='log', ylabel='Ratio')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')
    fig.savefig(output_file)


# HWP with instrumental noise
plot_comparison(cl_hwp_ins, 'ref', 'plots/hwp_instrumental_comparison.svg')

# HWP with atmospheric noise
plot_comparison(cl_hwp_atm, 'ref', 'plots/hwp_atmospheric_comparison.svg')

# No HWP with instrumental noise
plot_comparison(cl_nohwp_ins, 'ref', 'plots/nohwp_instrumental_comparison.svg')

# No HWP with atmospheric noise
plot_comparison(cl_nohwp_atm, 'ref', 'plots/nohwp_atmospheric_comparison.svg')


def plot_gap_filling_comparison(cl_dict, methods, output_file, ylabel=r'$N_\ell [\mu K^2]$'):
    """Generate plots comparing different gap filling methods against a reference."""
    fig, axs = plt.subplots(1, 3, sharex=True, layout='constrained', figsize=(10, 4))
    for i, ax in enumerate(axs):
        tt_ee_bb = ['TT', 'EE', 'BB'][i]
        ax.set(title=tt_ee_bb, xlabel=r'Multipole $\ell$')
        ax.axhline(y=1.0, c='k', ls='--')
        for method, label in methods.items():
            if method in cl_dict:
                ax.plot(
                    ell[ell_range],
                    cl_dict[method][tt_ee_bb][ell_range] / cl_dict['ref'][tt_ee_bb][ell_range],
                    label=label,
                )
    axs[0].set(yscale='log', ylabel=ylabel)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='outside upper center')
    fig.savefig(output_file)


# Define gap filling methods with their labels
gap_filling_methods = {
    'marg': 'original',
    'marg-real': 'realistic',
    'marg-noise-only': 'noise only',
}

# Plot for NoHWP with instrumental noise
plot_gap_filling_comparison(
    cl_nohwp_ins, gap_filling_methods, 'plots/nohwp_gap_filling_comparison.svg'
)

# Plot for HWP with atmospheric noise (note: marg-noise-only not available here)
plot_gap_filling_comparison(
    cl_hwp_atm,
    {'marg': 'original', 'marg-real': 'realistic'},
    'plots/hwp_gap_filling_comparison.svg',
)
