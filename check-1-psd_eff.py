#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils_mappraiser as utils

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight'})


# Simulation parameters
realization = 657
telescope = 0
component = 0
sindx = 0
detindx = 0
fsamp = 37
samples = 100000

freq = np.fft.rfftfreq(samples, 1 / fsamp)
npsd = len(freq) - 1

# PSD model
# log(sigma) = -4.82, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
sigma2 = 10**-4.82
alpha = 2.87
fknee = 1.05
fmin = 8.84e-4

# PSD model / effective
psd = utils.psd_model(freq, sigma2, alpha, fknee, fmin)
ipsd = 1 / psd
# psd[0] = 0


# cutoff using inv_tt
def cutoff(lambd, level: float = -3):
    """
    Find the approximate frequency at which the effective PSD has deviated
    from the model at the given level (in dB).
    """
    itt_model = utils.psd_to_ntt(ipsd, lambd)
    ipsd_eff = utils.autocorr_to_psd(itt_model, samples)
    power_ratio_dB = 10 * np.log10(np.reciprocal(ipsd_eff[1:]) / psd[1:])
    cutoff = freq[np.argmin(np.abs(power_ratio_dB - level)) + 1]
    return cutoff


fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey='col')

fig.suptitle(f'PSD and autocorrelation (TOD size = {samples})')
axs[0, 0].set_title('PSD model vs. effective')
axs[1, 0].set_title('Using inverse autocorrelation')
axs[2, 0].set_title('Using modified `tt`')

for ax in axs[:, 1]:
    # ax.set_title("Relative difference")
    ax.set_title('Attenuation / Amplification')
    ax.set_ylabel('dB')
    ax.axhline(y=0, c='dimgrey', label='model')
    ax.axhline(y=-3, ls=':', c='dimgrey', label='model -3 dB')

# Put labels on axes
for ax in axs[-1, :]:
    ax.set_xlabel(r'Frequency [$Hz$]')

for ax in axs[:, 0]:
    ax.set_ylabel(r'PSD [$K^2 / Hz$]')

# Plot references
for ax in axs[:, 0]:
    ax.loglog(freq[1:], psd[1:], c='k', label='model')

# Plots curves for different lambda values
lambdas = [4096, 8192, 16384, 32768]
cm = sns.color_palette('Set1', n_colors=len(lambdas))
for i, lambd in enumerate(lambdas):
    # PSD eff from `tt`
    tt_model = utils.psd_to_ntt(psd, lambd)
    psd_eff = utils.autocorr_to_psd(tt_model, samples)

    axs[0, 0].loglog(freq[1:], psd_eff[1:], c=cm[i], label=f'$\\lambda={lambd}$')
    axs[0, 1].semilogx(
        freq[1:],
        # (psd_eff - psd)[1:] / psd[1:],
        10 * np.log10(psd_eff[1:] / psd[1:]),
        c=cm[i],
        label=f'$\\lambda={lambd}$',
    )

    # PSD eff from `inv_tt`
    itt_model = utils.psd_to_ntt(ipsd, lambd)
    ipsd_eff = utils.autocorr_to_psd(itt_model, samples)

    axs[1, 0].loglog(freq[1:], 1 / ipsd_eff[1:], c=cm[i], label=f'$\\lambda={lambd}$')
    axs[1, 1].semilogx(
        freq[1:],
        # (1 / ipsd_eff - psd)[1:] / psd[1:],
        10 * np.log10(np.reciprocal(ipsd_eff[1:]) / psd[1:]),
        c=cm[i],
        label=f'$\\lambda={lambd}$',
    )
    axs[1, 1].axvline(x=cutoff(lambd), c=cm[i], ls=':')

    # PSD eff as `ifft(1/fft(inv_tt))`
    tt_modified = utils.psd_to_ntt(np.reciprocal(ipsd_eff), lambd)
    psd_eff_mod = utils.autocorr_to_psd(tt_modified, samples)

    axs[2, 0].loglog(freq[1:], psd_eff_mod[1:], c=cm[i], label=f'$\\lambda={lambd}$')
    axs[2, 1].semilogx(
        freq[1:],
        # (psd_eff_mod - psd)[1:] / psd[1:],
        10 * np.log10(psd_eff_mod[1:] / psd[1:]),
        c=cm[i],
        label=f'$\\lambda={lambd}$',
    )

    # Plot a dashed line at `fsamp/lambda`
    for ax in axs.flat:
        ax.axvline(x=fsamp / lambd, c=cm[i], ls='--')

for ax in axs.flat:
    ax.legend()
    ax.grid(True)

fig.tight_layout()
fig.savefig('plots/psd_effective.svg')
