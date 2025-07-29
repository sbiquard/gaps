#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

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

# Measure PSD of generated timestreams
NREAL = 25

tods = {
    'ins': {
        # 'py_ntt': np.zeros((NREAL, SAMPLES)),
        'c_ntt': np.zeros((NREAL, SAMPLES)),
        'toast_psd': np.zeros((NREAL, SAMPLES)),
        # 'py_psd': np.zeros((NREAL, SAMPLES)),
    },
    'atm': {
        # 'py_ntt': np.zeros((NREAL, SAMPLES)),
        'c_ntt': np.zeros((NREAL, SAMPLES)),
        'toast_psd': np.zeros((NREAL, SAMPLES)),
        # 'py_psd': np.zeros((NREAL, SAMPLES)),
    },
}
for model in ['ins', 'atm']:
    for i_ax in trange(NREAL):
        params = {
            'samples': SAMPLES,
            'realization': REALIZATION + i_ax * 6513754,
            'detindx': DETECTOR_INDEX,
            'sindx': SESSION_INDEX,
            'telescope': TELESCOPE,
            'fsamp': FSAMP,
            # 'verbose': i == 0,
        }
        # tods[model]['py_ntt'][i] = utils.sim_noise(
        #     py=True,
        #     autocorr=utils.psd_to_ntt(psd[model], LAGMAX),
        #     **params,
        # )
        tods[model]['c_ntt'][i_ax] = utils.sim_noise(
            autocorr=utils.psd_to_ntt(psd[model], LAGMAX),
            **params,
        )
        tods[model]['toast_psd'][i_ax] = utils.sim_noise(
            use_toast=True,
            freq=freq,
            psd=psd[model],
            **params,
        )
        # tods[model]['py_psd'][i] = utils.sim_noise(
        #     py=True,
        #     psd=psd[model],
        #     **params,
        # )

fig, axs = plt.subplots(2, 1, figsize=(10, 6), layout='constrained', sharex=True)

# First row for 'ins' timestreams
# (line1,) = axs[0].plot(tods['ins']['py_ntt'][0])
(line2,) = axs[0].plot(tods['ins']['c_ntt'][0])
(line3,) = axs[0].plot(tods['ins']['toast_psd'][0])
# (line4,) = axs[0].plot(tods['ins']['py_psd'][0])
axs[0].set_ylabel('Amplitude [arb. unit]')

# Second row for 'atm' timestreams
# axs[1].plot(tods['atm']['py_ntt'][0])
axs[1].plot(tods['atm']['c_ntt'][0])
axs[1].plot(tods['atm']['toast_psd'][0])
# axs[1].plot(tods['atm']['py_psd'][0])
axs[1].set_xlabel('Sample number')
axs[1].set_ylabel('Amplitude [arb. unit]')

# Add figure-level legend instead of per-subplot legends
fig.legend(
    [line2, line3],
    ['TOAST (uses PSD)', 'MAPPRAISER (uses autocorrelation)'],
    loc='outside upper center',
    ncol=2,
)
fig.savefig('plots/generated_tods.svg')


# Initialize dictionaries for the PSDs
psd_realizations = {
    model: {method: np.empty((NREAL, freq.size)) for method in tods[model].keys()}
    for model in tods.keys()
}

fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout='constrained', sharey=True)
axs[0].set_title('Instrumental')
axs[1].set_title('Atmospheric')

axs[0].set_ylabel('Ratio to model [dB]')
for ax in axs:
    ax.set(xlabel='Frequency [Hz]')
    ax.grid(True)

# Create common legend handles and labels
legend_handles = []

cm = sns.color_palette(n_colors=2)
for i_ax, (model, psd_real) in enumerate(psd_realizations.items()):
    ax = axs[i_ax]

    # Add model line to both axes and collect for legend
    model_line = ax.axhline(y=0, c='k', ls='--')  # 0 dB reference line
    if i_ax == 0:
        legend_handles.append(model_line)

    for i_method, method in enumerate(['c_ntt', 'toast_psd']):
        for j in trange(NREAL):
            fit_params = utils.fit_psd_to_tod(tods[model][method][j], FSAMP, welch=True)
            psd_real[method][j] = utils.psd_model(freq, *fit_params)

        avg_psd1 = np.average(psd_real[method], axis=0)[1:]
        dev_psd1 = np.std(psd_real[method], axis=0)[1:]

        # Convert ratios to decibels
        ratio_db = 10 * np.log10(avg_psd1 / psd1[model])
        upper_db = 10 * np.log10((avg_psd1 + dev_psd1) / psd1[model])
        lower_db = 10 * np.log10((avg_psd1 - dev_psd1) / psd1[model])

        # Plot line and shaded region for one sigma
        (line,) = ax.semilogx(freq1, ratio_db, color=cm[i_method])
        ax.fill_between(freq1, lower_db, upper_db, color=cm[i_method], alpha=0.3)

        # Only add to legend once
        if i_ax == 0:
            legend_handles.append(line)

# Create a single legend above the figure
fig.legend(
    legend_handles,
    ['Model', 'TOAST', 'MAPPRAISER'],
    loc='outside upper center',
    ncol=3,
)

fig.savefig('plots/psd_fits_vs_model.svg')
