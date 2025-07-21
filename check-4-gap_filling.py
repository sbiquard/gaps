#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import utils_mappraiser as utils

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 150, 'savefig.bbox': 'tight'})


# parameters
realization = 657
telescope = 0
component = 0
sindx = 0
detindx = 0
sample_rate = 37
samples = 100000

# log(sigma) = -4.82, alpha = -0.87, fknee = 0.05, fmin = 8.84e-04
sigma2 = 10**-4.82
alpha = 2.87
fknee = 1.05
fmin = 8.84e-4

# form the input PSD model
freq = np.fft.rfftfreq(samples, 1 / sample_rate)
npsd = len(freq) - 1

psd_model = utils.psd_model(freq, sigma2, alpha, fknee, fmin)
# ipsd = utils.inversepsd_model(freq, 1 / sigma2, alpha, fknee, fmin)
psd_model[0] = 0

tod_toast = utils.sim_noise(
    samples=samples,
    realization=realization,
    detindx=detindx,
    sindx=sindx,
    telescope=telescope,
    fsamp=sample_rate,
    use_toast=True,
    freq=freq,
    psd=psd_model,
)

fit_params = utils.fit_psd_to_tod(tod_toast, sample_rate, welch=False)
psd_fit = utils.psd_model(freq, *fit_params)
ipsd_fit = np.reciprocal(psd_fit)

lagmax = 2**16
invntt = utils.psd_to_invntt(psd_fit, lagmax)
ntt = utils.effective_ntt(invntt, utils.next_fast_fft_size(samples))

# plt.figure()
# plt.title('Power spectral densities')
# plt.loglog(freq[1:], psd_model[1:], c='k', label='model')
# plt.loglog(freq[1:], psd_fit[1:], label='fit')
# plt.loglog(freq[1:], utils.compute_psd(tt_bis, samples)[1:], label='eff')
# plt.xlabel('frequency [$Hz$]')
# plt.ylabel('PSD in $[tod]^2 / Hz$')
# plt.grid(True)
# plt.legend()
# plt.show()

# ____________________________________________________________
# 4) Introduce some gaps


def get_pix(nsamp, lgap=2**12, step=samples // 5, return_all=False, verbose=True):
    pix = np.ones(nsamp, dtype=np.int32)
    valid = np.ones(nsamp, dtype=np.uint8)
    for i in range(0, nsamp, step):
        slc = slice(i, i + lgap)
        pix[slc] = -1
        valid[slc] = 0
    if verbose:
        print(f'Valid samples: {np.sum(valid)}/{samples} = {100 * np.sum(valid) / samples} %')
    if return_all:
        return pix, valid, lgap, step
    else:
        return pix


pix, valid, lgap, step = get_pix(samples, return_all=True)


# ____________________________________________________________
# 5) Perform gap filling

NREAL = 5
FACTORS = [1, 2, 4]
tods_filled = {
    'fsamp': np.empty((NREAL, samples)),
    'fact_1': np.empty((NREAL, samples)),
    'fact_2': np.empty((NREAL, samples)),
    'fact_4': np.empty((NREAL, samples)),
}

for i in tqdm(range(NREAL)):
    for fact in FACTORS:
        tods_filled[f'fact_{fact}'][i, :] = utils.sim_constrained_block(
            False,  # initialize MPI
            False,  # finalize MPI
            lagmax // fact,
            tod_toast,
            ntt,
            invntt,
            pix,
            fsamp=sample_rate,
            realization=realization + i + 1,
            detindx=detindx,
            sindx=sindx,
            telescope=telescope,
        )
    tods_filled['fsamp'][i, :] = utils.sim_constrained_block(
        False,  # initialize MPI
        False,  # finalize MPI
        int(sample_rate),
        tod_toast,
        ntt,
        invntt,
        pix,
        fsamp=sample_rate,
        realization=realization + i + 1,
        detindx=detindx,
        sindx=sindx,
        telescope=telescope,
    )


plt.figure(figsize=(10, 5))
plt.title(f'length={samples} - lambda={lagmax} - gaps: {lgap} every {step} samples')

# original timestream
plt.plot(range(samples), (tod_toast - tod_toast), 'k', label='ref')
# gap-filled timestreams
plt.plot(range(samples), (tods_filled['fsamp'][0] - tod_toast), label=r'$\Delta w = f_s$')
plt.plot(range(samples), (tods_filled['fact_1'][0] - tod_toast), label=r'$\Delta w = 2\lambda$')
plt.plot(range(samples), (tods_filled['fact_2'][0] - tod_toast), label=r'$\Delta w = \lambda$')
plt.plot(range(samples), (tods_filled['fact_4'][0] - tod_toast), label=r'$\Delta w = \lambda / 2$')

# gap edges
for i in range(valid.size - 1):
    if i == 0 and valid[1] == 0:
        plt.axvline(x=i, ls=':', c='k')
    elif i == valid.size - 1 and valid[i] == 0:
        plt.axvline(x=i + 1, ls=':', c='k')
    elif (valid[i] + valid[i + 1]) == 1:
        # change between i and i + 1
        plt.axvline(x=0.5 + i, ls=':', c='k')

# plt.yscale('symlog')
# plt.xlim(right=int(lgap * 1.1))
plt.legend()
plt.savefig('plots/filled_tods.svg')

# periodograms

psds_filled = {
    'fsamp': np.empty((NREAL, len(psd_fit)), dtype=psd_fit.dtype),
    'fact_1': np.empty((NREAL, len(psd_fit)), dtype=psd_fit.dtype),
    'fact_2': np.empty((NREAL, len(psd_fit)), dtype=psd_fit.dtype),
    'fact_4': np.empty((NREAL, len(psd_fit)), dtype=psd_fit.dtype),
}

for i in range(NREAL):
    for key in ['fsamp', 'fact_1', 'fact_2', 'fact_4']:
        fit_params = utils.fit_psd_to_tod(tods_filled[key][i], sample_rate, welch=False)
        psds_filled[key][i] = utils.psd_model(freq, *fit_params)

# Create dictionaries for PSD averages and standard deviations
psd_avg = {
    'fsamp': np.mean(psds_filled['fsamp'], axis=0),
    'fact_1': np.mean(psds_filled['fact_1'], axis=0),
    'fact_2': np.mean(psds_filled['fact_2'], axis=0),
    'fact_4': np.mean(psds_filled['fact_4'], axis=0),
}
psd_dev = {
    'fsamp': np.std(psds_filled['fsamp'], axis=0),
    'fact_1': np.std(psds_filled['fact_1'], axis=0),
    'fact_2': np.std(psds_filled['fact_2'], axis=0),
    'fact_4': np.std(psds_filled['fact_4'], axis=0),
}

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex='all')
fig.suptitle('Before / after comparison')

axs[0].set_title('fitted PSD')
axs[0].loglog(freq[1:], psd_fit[1:], 'k--', label='before')
axs[0].set_ylabel('$[tod]^2 / Hz$')

axs[1].set_title('ratio in dB')
axs[1].axhline(y=0.0, c='k', ls='--', label='before')
axs[1].set_ylabel('$dB$')

for key, lab in zip(
    ('fsamp', 'fact_1', 'fact_2', 'fact_4'),
    (
        r'$\Delta w = f_s$',
        r'$\Delta w = 2\lambda$',
        r'$\Delta w = \lambda$',
        r'$\Delta w = \lambda / 2$',
    ),
):
    p0 = axs[0].loglog(freq[1:], psd_avg[key][1:], label=lab)
    axs[0].loglog(freq[1:], (psd_avg[key] - psd_dev[key])[1:], ls=':', c=p0[0].get_color())
    axs[0].loglog(freq[1:], (psd_avg[key] + psd_dev[key])[1:], ls=':', c=p0[0].get_color())
    p1 = axs[1].semilogx(freq[1:], 10 * np.log10(psd_avg[key][1:] / psd_fit[1:]), label=lab)
    axs[1].semilogx(
        freq[1:],
        10 * np.log10((psd_avg[key] - psd_dev[key])[1:] / psd_fit[1:]),
        ls=':',
        c=p1[0].get_color(),
    )
    axs[1].semilogx(
        freq[1:],
        10 * np.log10((psd_avg[key] + psd_dev[key])[1:] / psd_fit[1:]),
        ls=':',
        c=p1[0].get_color(),
    )

for ax in axs.flat:
    ax.set_xlabel('frequency [$Hz$]')
    ax.grid(True)
    ax.legend()

fig.tight_layout()
fig.savefig('plots/gap_filling_consistency.svg')
