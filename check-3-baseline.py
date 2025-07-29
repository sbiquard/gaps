#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import welch

import utils_mappraiser as utils

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight'})


# parameters
REALIZATION = 657
TELESCOPE = 0
COMPONENT = 0
SESSION_INDEX = 0
DETECTOR_INDEX = 0
FSAMP = 37
SAMPLES = 2**17
LAGMAX = 2**15
LGAP = LAGMAX // 8
STEP = LAGMAX // 2
OFFSET = LAGMAX // 16
W0_VALUES = [LAGMAX // n for n in (16, 4, 2, 1)]


freq = np.fft.rfftfreq(SAMPLES, 1 / FSAMP)
freq1 = freq[1:]  # Without zero frequency
npsd = len(freq)

# PSD model
# log(sigma) = -4.82, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
SIGMA = 1.0
ALPHA_ATM = 2.5
# alpha_ins = 1.0
FKNEE_ATM = 1.0
# fknee_ins = 0.05
FMIN = 1e-3

psd = utils.psd_model(freq, SIGMA, ALPHA_ATM, FKNEE_ATM, FMIN)
tod = utils.sim_noise(
    samples=SAMPLES,
    realization=REALIZATION,
    detindx=DETECTOR_INDEX,
    sindx=SESSION_INDEX,
    telescope=TELESCOPE,
    fsamp=FSAMP,
    use_toast=True,
    freq=freq,
    psd=psd,
)

fit_params_tod = utils.fit_psd_to_tod(tod, FSAMP)

psd_fit = utils.psd_model(freq, *fit_params_tod)
ipsd_fit = np.reciprocal(psd_fit)


pix = np.ones(SAMPLES, dtype=np.int32)
valid = np.ones(SAMPLES, dtype=np.uint8)
for i in range(OFFSET, SAMPLES, STEP):
    slc = slice(i, i + LGAP)
    pix[slc] = -1
    valid[slc] = 0

print(f'Gaps of length {LGAP} every {STEP} samples starting at {OFFSET}')
print(f'Valid samples: {np.sum(valid) / SAMPLES:%}')
print(f'{W0_VALUES=}')


def plot_gap_edges(valid, ax, ls='dotted', c='k'):
    for i in range(0, len(valid) - 1):
        if i == 0 and valid[1] == 0:
            ax.axvline(x=i, ls=ls, c=c)
        elif i == len(valid) - 1 and valid[i] == 0:
            ax.axvline(x=i + 1, ls=ls, c=c)
        elif (valid[i] + valid[i + 1]) == 1:
            # change between i and i + 1
            ax.axvline(x=0.5 + i, ls=ls, c=c)


baselines = {}
tods_rm = {}
for w0 in W0_VALUES:
    baselines[w0], tods_rm[w0] = utils.baseline(np.array(tod), valid, w0, cp=True)

# plot baselines + TODs after removal
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True, layout='constrained')
plot_gap_edges(valid, axs[0])
plot_gap_edges(valid, axs[1])

# First subplot - baselines comparison
axs[0].plot(tod, c='k', label='TOD')
for w0 in W0_VALUES:
    axs[0].plot(baselines[w0], ls='--', label=f'$\\Delta w={2 * w0 + 1}$')

# Second subplot - TODs after baseline removal
for w0 in W0_VALUES:
    axs[1].plot(tods_rm[w0])

# Get handles and labels from the first subplot and place legend above the figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncols=len(handles))

fig.savefig('plots/baselines_before_after.svg')


# _________________________________________________________________________________________________
# Baseline power spectra

fig, ax = plt.subplots()

fit_params_tod, f_tod, psd_tod = utils.fit_psd_to_tod(
    tod, FSAMP, welch=False, return_periodogram=True
)
ax.loglog(f_tod[1:], utils.psd_model(f_tod[1:], *fit_params_tod), 'k--', label='PSD of timestream')

for i, w0 in enumerate(W0_VALUES):
    f, psd = welch(baselines[w0], fs=FSAMP, nperseg=2 * LAGMAX)
    ax.loglog(f[1:], psd[1:], label=f'$\\Delta w={2 * w0 + 1}$')
ax.legend()
ax.set_ylim(bottom=1e-4)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('PSD [arb. unit]')
ax.grid(True)
fig.savefig('plots/baseline_power_spectra.svg')


# _________________________________________________________________________________________________
# Effect on TOD power spectra


def cutoff_dma(w0):
    wsize = 2 * w0 + 1
    return 0.442947 / np.sqrt(wsize**2 - 1)


def plot_baseline_removal(w0, ax=None, plot_bline_fit=False):
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(f'$\\Delta w={2 * w0 + 1}$')

    # Compute the baseline
    bline, new_tod = utils.baseline(tod, valid, w0, cp=True)

    # Original TOD
    ax.loglog(f_tod[1:], utils.psd_model(f_tod[1:], *fit_params_tod), 'k--', label='PSD model')
    # ax.loglog(f_tod[1:], np.convolve(psd_tod, kernel, mode='same')[1:], c='k', alpha=0.5)
    ax.loglog(f_tod[1:], psd_tod[1:], c='k', alpha=0.6)

    # TOD after removal
    fit_params_removal, f, psd = utils.fit_psd_to_tod(
        new_tod,
        FSAMP,
        welch=False,
        return_periodogram=True,
    )
    ax.loglog(f[1:], utils.psd_model(f[1:], *fit_params_removal), 'b--', label='PSD after removal')
    # ax.loglog(f[1:], np.convolve(psd, kernel, mode='same')[1:], c='b', alpha=0.5)
    ax.loglog(f[1:], psd[1:], c='b', alpha=0.6)

    # ax.axvline(x=sample_rate/(2*w0+1), c='r', label="$\Delta w/f_s$")
    ax.axvline(x=FSAMP * cutoff_dma(w0), c='r', label='-3 dB cutoff')

    # Baseline
    fit_params_baseline, f, psd = utils.fit_psd_to_tod(
        bline,
        FSAMP,
        welch=False,
        return_periodogram=True,
    )
    if plot_bline_fit:
        ax.loglog(
            f[1:],
            utils.psd_model(f[1:], *fit_params_baseline),
            ls='--',
            c='orange',
            label='baseline PSD',
        )
        # ax.loglog(f[1:], np.convolve(psd, kernel, mode='same')[1:], c='orange', alpha=0.5)
        ax.loglog(f[1:], psd[1:], c='orange', alpha=0.6)
    else:
        ax.loglog(
            f[1:],
            # np.convolve(psd, kernel, mode='same')[1:],
            psd[1:],
            c='orange',
            alpha=0.6,
            label='baseline periodogram',
        )


fig, axs = plt.subplots(
    2, len(W0_VALUES) // 2, figsize=(8, 8), sharex=True, sharey=True, layout='constrained'
)
for i, w0 in enumerate(W0_VALUES):
    ax = axs.flat[i]
    plot_baseline_removal(w0, ax=ax)
    ax.set_ylim(bottom=1e-2)
    ax.grid(True)
    ax.set(xlabel='Frequency [Hz]', ylabel='PSD [arb. unit]')
    ax.label_outer()

fig.legend(
    *axs.flat[0].get_legend_handles_labels(),
    loc='outside upper center',
    ncols=5,
)
fig.savefig('plots/baseline_removal_psd_comparison_eff.svg')
