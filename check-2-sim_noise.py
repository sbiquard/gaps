#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

import utils_mappraiser as utils

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# parameters
realization = 657
telescope = 0
component = 0
sindx = 0
detindx = 0
fsamp = 37
samples = 100000

# PSD fit log(sigma2) = -9.65, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
# sigma2 = 10**(-9.65)
sigma2 = 1
alpha = 2.87
fknee = 1.05
fmin = 8.84e-4

# Generate timestream
freq = np.fft.rfftfreq(samples, 1 / fsamp)
psd = utils.psd_model(freq, sigma2, alpha, fknee, fmin)
psd[0] = 0

fftlen = utils.next_fast_fft_size(samples)
print(f'{fftlen=}')

freq_in = np.fft.rfftfreq(fftlen, 1 / fsamp)
psd_in = utils.psd_model(freq_in, sigma2, alpha, fknee, fmin)
psd_in[0] = 0

# correlation length
lcorr = 2**16

# Measure PSD of generated timestreams
n_real = 100

tods = {
    'py_ntt': np.zeros((n_real, samples)),
    'c_ntt': np.zeros((n_real, samples)),
    'toast_psd': np.zeros((n_real, samples)),
    'py_psd': np.zeros((n_real, samples)),
}

for i in trange(n_real):
    params = {
        'samples': samples,
        'realization': realization + i * 6513754,
        'detindx': detindx,
        'sindx': sindx,
        'telescope': telescope,
        'fsamp': fsamp,
        # 'verbose': i == 0,
    }
    tods['py_ntt'][i] = utils.sim_noise(
        py=True,
        autocorr=utils.psd_to_ntt(psd, lcorr),
        **params,
    )
    tods['c_ntt'][i] = utils.sim_noise(
        autocorr=utils.psd_to_ntt(psd, lcorr),
        **params,
    )
    tods['toast_psd'][i] = utils.sim_noise(
        use_toast=True,
        freq=freq,
        psd=psd,
        **params,
    )
    tods['py_psd'][i] = utils.sim_noise(
        py=True,
        psd=psd_in,
        **params,
    )

plt.figure(figsize=(12, 8))

# First subplot for timestreams
plt.subplot(2, 1, 1)
plt.title('Generated timestreams')
plt.plot(tods['py_ntt'][0], label='Py - using Ntt')
plt.plot(tods['c_ntt'][0], label='C - using Ntt')
plt.plot(tods['toast_psd'][0], label='TOAST - using PSD')
plt.plot(tods['py_psd'][0], label='Py - using PSD')
plt.legend(loc='upper right')

# Second subplot for measured PSDs
plt.subplot(2, 1, 2)
plt.title('Measured PSDs')

# Fit parameters for py_ntt and plot
py_ntt_params = utils.fit_psd_to_tod(tods['py_ntt'][0], fsamp)[0]
plt.loglog(
    freq[1:],
    utils.psd_model(freq[1:], *py_ntt_params),
    label='Py - using Ntt',
)
print(
    f'Py-Ntt fitted params: sigma2={py_ntt_params[0]:.4e}, alpha={py_ntt_params[1]:.4f}, fknee={py_ntt_params[2]:.4f}, fmin={py_ntt_params[3]:.4e}'
)

# Fit parameters for c_ntt and plot
c_ntt_params = utils.fit_psd_to_tod(tods['c_ntt'][0], fsamp)[0]
plt.loglog(
    freq[1:],
    utils.psd_model(freq[1:], *c_ntt_params),
    label='C - using Ntt',
)
print(
    f'C-Ntt fitted params: sigma2={c_ntt_params[0]:.4e}, alpha={c_ntt_params[1]:.4f}, fknee={c_ntt_params[2]:.4f}, fmin={c_ntt_params[3]:.4e}'
)

# Fit parameters for toast_psd and plot
toast_psd_params = utils.fit_psd_to_tod(tods['toast_psd'][0], fsamp)[0]
plt.loglog(
    freq[1:],
    utils.psd_model(freq[1:], *toast_psd_params),
    label='TOAST - using PSD',
)
print(
    f'TOAST-PSD fitted params: sigma2={toast_psd_params[0]:.4e}, alpha={toast_psd_params[1]:.4f}, fknee={toast_psd_params[2]:.4f}, fmin={toast_psd_params[3]:.4e}'
)

# Fit parameters for py_psd and plot
py_psd_params = utils.fit_psd_to_tod(tods['py_psd'][0], fsamp)[0]
plt.loglog(
    freq[1:],
    utils.psd_model(freq[1:], *py_psd_params),
    label='Py - using PSD',
)
print(
    f'Py-PSD fitted params: sigma2={py_psd_params[0]:.4e}, alpha={py_psd_params[1]:.4f}, fknee={py_psd_params[2]:.4f}, fmin={py_psd_params[3]:.4e}'
)

# Plot the model
plt.loglog(freq[1:], psd[1:], 'k--', label='Model')
print(f'Original model: sigma2={sigma2:.4e}, alpha={alpha:.4f}, fknee={fknee:.4f}, fmin={fmin:.4e}')

plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plots/generated_tods.png')

# Initialize dictionaries for the PSDs
psds_simple = {key: np.empty((n_real, freq.size)) for key in tods.keys()}
psds_hann = {key: np.empty((n_real, freq.size)) for key in tods.keys()}

for i in trange(n_real):
    for key in tods.keys():
        # Simple periodogram PSDs with Hann window
        fit_params_simple = utils.fit_psd_to_tod(tods[key][i], fsamp, welch=False)[0]
        psds_simple[key][i] = utils.psd_model(freq, *fit_params_simple)

        # Welch PSDs
        fit_params_welch = utils.fit_psd_to_tod(tods[key][i], fsamp, welch=True)[0]
        psds_hann[key][i] = utils.psd_model(freq, *fit_params_welch)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# fig.suptitle(f'PSD average fits (n={n_real}) compared to model')
axs[0].set_title('Simple periodogram')
axs[1].set_title('Welch method')

# Create common legend handles and labels
legend_handles = []
legend_labels = []

# Add model line to both axes and collect for legend
axs[0].axhline(y=1, c='k', ls='--')
model_line = axs[1].axhline(y=1, c='k', ls='--')
legend_handles.append(model_line)
legend_labels.append('Model')

for ax in axs:
    ax.set(xlabel='Frequency [Hz]', ylabel='Ratio to model')

cm = sns.color_palette('Set1')
for i, (key, label) in enumerate(
    zip(
        tods.keys(),
        ['Py - using Ntt', 'C - using Ntt', 'TOAST - using PSD', 'Py - using PSD'],
    )
):
    avg_s = np.average(psds_simple[key], axis=0)
    dev_s = np.std(psds_simple[key], axis=0)
    avg_h = np.average(psds_hann[key], axis=0)
    dev_h = np.std(psds_hann[key], axis=0)

    # Plot in first axis
    line_s, = axs[0].semilogx(freq[1:], avg_s[1:] / psd[1:], c=cm[i])
    axs[0].semilogx(freq[1:], (avg_s - dev_s)[1:] / psd[1:], c=cm[i], ls=':')
    axs[0].semilogx(freq[1:], (avg_s + dev_s)[1:] / psd[1:], c=cm[i], ls=':')

    # Plot in second axis
    axs[1].semilogx(freq[1:], avg_h[1:] / psd[1:], c=cm[i])
    axs[1].semilogx(freq[1:], (avg_h - dev_h)[1:] / psd[1:], c=cm[i], ls=':')
    axs[1].semilogx(freq[1:], (avg_h + dev_h)[1:] / psd[1:], c=cm[i], ls=':')

    # Only add to legend once
    legend_handles.append(line_s)
    legend_labels.append(label)

# Create a single legend above the figure
fig.legend(
    legend_handles,
    legend_labels,
    loc='outside upper center',
    ncol=5,
)

# fig.tight_layout()
# fig.subplots_adjust(top=0.85)  # Make room for the legend above
fig.savefig('plots/psd_fits_vs_model.svg')
