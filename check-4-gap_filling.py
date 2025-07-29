#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

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
LAGMAX = 2**16
LGAP = LAGMAX // 8
STEP = LAGMAX // 2
OFFSET = LAGMAX // 16
# W0_VALUES = [LAGMAX // n for n in (4, 2, 1)]
NREAL = 5

# PSD model
# log(sigma) = -4.82, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
SIGMA = 1.0
ALPHA_ATM = 2.5
ALPHA_INS = 1.0
FKNEE_ATM = 1.0
FKNEE_INS = 0.05
FMIN = 1e-3

freq = np.fft.rfftfreq(SAMPLES, 1 / FSAMP)
freq1 = freq[1:]  # Without zero frequency
npsd = len(freq)

psd = {
    'ins': utils.psd_model(freq, SIGMA, ALPHA_INS, FKNEE_INS, FMIN),
    'atm': utils.psd_model(freq, SIGMA, ALPHA_ATM, FKNEE_ATM, FMIN),
}
# ipsd = utils.inversepsd_model(freq, 1 / sigma2, alpha, fknee, fmin)
# psd_atm[0] = 0

tod = {
    k: utils.sim_noise(
        samples=SAMPLES,
        realization=REALIZATION,
        detindx=DETECTOR_INDEX,
        sindx=SESSION_INDEX,
        telescope=TELESCOPE,
        fsamp=FSAMP,
        use_toast=True,
        freq=freq,
        psd=_psd,
    )
    for k, _psd in psd.items()
}

fit_params = {k: utils.fit_psd_to_tod(_tod, FSAMP) for k, _tod in tod.items()}
psd_fit = {k: utils.psd_model(freq, *_params) for k, _params in fit_params.items()}
ipsd_fit = {k: 1 / v for k, v in psd_fit.items()}
invntt = {k: utils.psd_to_invntt(_psd_fit, LAGMAX) for k, _psd_fit in psd_fit.items()}
ntt = {k: utils.effective_ntt(_invntt, SAMPLES) for k, _invntt in invntt.items()}

# ____________________________________________________________
# 4) Introduce some gaps


PIX = np.ones(SAMPLES, dtype=np.int32)
VALID = np.ones(SAMPLES, dtype=np.uint8)
for i in range(OFFSET, SAMPLES, STEP):
    slc = slice(i, i + LGAP)
    PIX[slc] = -1
    VALID[slc] = 0

print(f'Gaps of length {LGAP} every {STEP} samples starting at {OFFSET}')
print(f'Valid samples: {np.sum(VALID) / SAMPLES:%}')


def plot_gap_edges(ax, ls='dotted', c='k'):
    for i in range(0, len(VALID) - 1):
        if i == 0 and VALID[1] == 0:
            ax.axvline(x=i, ls=ls, c=c)
        elif i == len(VALID) - 1 and VALID[i] == 0:
            ax.axvline(x=i + 1, ls=ls, c=c)
        elif (VALID[i] + VALID[i + 1]) == 1:
            # change between i and i + 1
            ax.axvline(x=0.5 + i, ls=ls, c=c)


# ____________________________________________________________
# 5) Perform gap filling

tods_filled = {k: np.empty((NREAL, SAMPLES)) for k in ['atm', 'ins']}

for i in tqdm(range(NREAL)):
    for k, _tods in tods_filled.items():
        _tods[i] = utils.sim_constrained_block(
            False,  # initialize MPI
            False,  # finalize MPI
            -1,
            tod[k],
            ntt[k],
            invntt[k],
            PIX,
            fsamp=FSAMP,
            realization=REALIZATION + i + 1,
            detindx=DETECTOR_INDEX,
            sindx=SESSION_INDEX,
            telescope=TELESCOPE,
        )

fig, axs = plt.subplots(2, 1, figsize=(8, 4), layout='constrained', sharex=True)
axs[0].set_title('Instrumental')
axs[1].set_title('Atmospheric')
axs[0].plot(tod['ins'], 'k')
axs[1].plot(tod['atm'], 'k')
for i in range(min(5, NREAL)):
    axs[0].plot(tods_filled['ins'][i])
    axs[1].plot(tods_filled['atm'][i])
for ax in axs:
    plot_gap_edges(ax)
    ax.set_xlim(0, 50_000)
    ax.set_ylabel('TOD amplitude [arb. unit]')
axs[-1].set_xlabel('Sample number')
fig.savefig('plots/filled_tods.svg')

# periodograms

psds_filled = {k: np.empty((NREAL, npsd)) for k in ['ins', 'atm']}

for i in range(NREAL):
    for k, _tods in tods_filled.items():
        psds_filled[k][i] = utils.psd_model(
            freq, *utils.fit_psd_to_tod(_tods[i], FSAMP, welch=False)
        )

# Create dictionaries for PSD averages and standard deviations
psd_avg = {k: np.mean(v, axis=0) for k, v in psds_filled.items()}
psd_dev = {k: np.std(v, axis=0) for k, v in psds_filled.items()}

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, layout='constrained')
axs[0].set_ylabel('PSD [arb. unit]')
axs[1].set_ylabel('Ratio [dB]')
for ax in axs:
    ax.grid(True)
    ax.set_xlabel('Frequency [Hz]')
cm = sns.color_palette('Dark2', n_colors=2)
for ik, k in enumerate(['ins', 'atm']):
    axs[0].loglog(freq1, psd[k][1:], c='k', ls='--', label='model' if k == 'ins' else None)
    axs[0].loglog(
        freq1, psd_avg[k][1:], c=cm[ik], label='Instrumental' if k == 'ins' else 'Atmospheric'
    )
    axs[0].fill_between(
        freq1,
        psd_avg[k][1:] - psd_dev[k][1:],
        psd_avg[k][1:] + psd_dev[k][1:],
        color=cm[ik],
        alpha=0.5,
    )
    axs[1].axhline(y=0.0, c='k', ls='--')
    axs[1].semilogx(freq1, 10 * np.log10(psd_avg[k] / psd[k])[1:], c=cm[ik])
    axs[1].fill_between(
        freq1,
        10 * np.log10((psd_avg[k] - psd_dev[k]) / psd[k])[1:],
        10 * np.log10((psd_avg[k] + psd_dev[k]) / psd[k])[1:],
        color=cm[ik],
        alpha=0.5,
    )
fig.legend(loc='outside upper center', ncol=4)
fig.savefig('plots/gap_filling_consistency.svg')
