#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils_mappraiser as utils

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight'})


# Simulation parameters
REALIZATION = 657
TELESCOPE = 0
COMPONENT = 0
SESSION_INDEX = 0
DETECTOR_INDEX = 0
FSAMP = 37
SAMPLES = 2**17

freq = np.fft.rfftfreq(SAMPLES, 1 / FSAMP)
freq1 = freq[1:]  # Without zero frequency
npsd = len(freq) - 1

# PSD model
# log(sigma) = -4.82, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
SIGMA = 1.0
ALPHA_ATM = 2.5
ALPHA_INS = 1.0
FKNEE_ATM = 1.0
FKNEE_INS = 0.05
FMIN = 1e-3

psd = {
    'ins': utils.psd_model(freq, SIGMA, ALPHA_INS, FKNEE_INS, FMIN),
    'atm': utils.psd_model(freq, SIGMA, ALPHA_ATM, FKNEE_ATM, FMIN),
}
psd1 = {k: v[1:] for k, v in psd.items()}
ipsd = {k: 1 / v for k, v in psd.items()}

# Plot comparing the two PSDs
fig, ax = plt.subplots()
ax.loglog(freq1, psd1['ins'], label='Instrumental')
ax.loglog(freq1, psd1['atm'], label='Atmospheric')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('PSD [arb. unit]')
ax.grid(True)
ax.legend()
fig.savefig('plots/psd_comparison.svg')


for model in ['ins', 'atm']:
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey='row', layout='constrained')

    axs[0, 0].set_title('Using direct autocorrelation')
    axs[0, 1].set_title('Using inverse autocorrelation')
    axs[0, 2].set_title('Using modified autocorrelation')

    # Create empty lists to store legend handles and labels
    legend_elements = []

    # Add horizontal lines for reference
    for ax in axs[1, :]:
        model_line = ax.axhline(y=0, c='dimgrey')
        model_db_line = ax.axhline(y=-3, ls=':', c='dimgrey')

    # Add these to legend elements
    legend_elements.append((model_line, 'model'))
    legend_elements.append((model_db_line, 'model -3 dB'))

    # Put labels on axes
    for ax in axs[1, :]:
        ax.set_xlabel('Frequency [Hz]')

    axs[0, 0].set_ylabel('PSD [arb. unit]')
    axs[1, 0].set_ylabel('Ratio over model [dB]')

    # Plot references
    for ax in axs[0, :]:
        ref_line = ax.loglog(freq1, psd1[model], c='k')[0]

    legend_elements.append((ref_line, 'model'))

    # Plots curves for different lambda values
    LAGS = [4096, 8192, 16384, 32768]
    cm = sns.color_palette('Set1', n_colors=len(LAGS))
    for i, lag in enumerate(LAGS):
        # PSD eff from `tt`
        ntt = utils.psd_to_ntt(psd[model], lag)
        psd1_eff = utils.autocorr_to_psd(ntt, SAMPLES)[1:]

        line1 = axs[0, 0].loglog(freq1, psd1_eff, c=cm[i])[0]
        axs[1, 0].semilogx(
            freq1,
            10 * np.log10(psd1_eff / psd1[model]),
            c=cm[i],
        )

        # PSD eff from `inv_tt`
        invntt = utils.psd_to_ntt(ipsd[model], lag)
        ipsd1_eff = utils.autocorr_to_psd(invntt, SAMPLES)[1:]

        axs[0, 1].loglog(freq1, 1 / ipsd1_eff, c=cm[i])
        axs[1, 1].semilogx(
            freq1,
            10 * np.log10(1 / ipsd1_eff / psd1[model]),
            c=cm[i],
        )
        # axs[1, 1].axvline(x=utils.cutoff(freq, psd[model], lag, SAMPLES), c=cm[i], ls=':')

        # PSD eff as `ifft(1/fft(inv_tt))`
        ntt_mod = utils.psd_to_ntt(np.reciprocal(ipsd1_eff), lag)
        psd1_eff_mod = utils.autocorr_to_psd(ntt_mod, SAMPLES)[1:]

        axs[0, 2].loglog(freq1, psd1_eff_mod, c=cm[i])
        axs[1, 2].semilogx(
            freq1,
            10 * np.log10(psd1_eff_mod / psd1[model]),
            c=cm[i],
        )

        # Plot a dashed line at `fsamp/lambda` and add to legend
        vline = None
        for ax in axs.flat:
            vline = ax.axvline(x=FSAMP / lag, c=cm[i], ls='--', lw=0.8)

        # Add this lambda value to legend elements
        legend_elements.append((line1, f'$\\lambda={lag}$'))

    # Create legend handles and labels from the stored elements
    handles, labels = zip(*legend_elements)
    fig.legend(handles, labels, loc='outside upper center', ncol=len(handles))

    for ax in axs.flat:
        ax.grid(True)

    fig.savefig(f'plots/psd_effective_{model}.svg')
