import numpy as np
import numpy.typing as npt
import scipy.signal
from scipy.optimize import curve_fit
from scipy.signal import get_window
from toast import rng
from toast.ops.sim_tod_noise import sim_noise_timestream

import mappraiser.wrapper as mappraiser


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a, strict=True)


def next_fast_fft_size(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


def interpolate_psd(
    freq: npt.NDArray, psd: npt.NDArray, fft_size: int, rate: float = 1.0
) -> npt.NDArray:
    """Perform a logarithmic interpolation of PSD values"""
    interp_freq = np.fft.rfftfreq(fft_size, 1 / rate)
    # shift by fixed amounts in frequency and amplitude to avoid zeros
    freq_shift = rate / fft_size
    psd_shift = 0.01 * np.min(np.where(psd > 0, psd, 0))
    log_x = np.log10(interp_freq + freq_shift)
    log_xp = np.log10(freq + freq_shift)
    log_fp = np.log10(psd + psd_shift)
    interp_psd = np.interp(log_x, log_xp, log_fp)
    return np.power(10.0, interp_psd) - psd_shift


def estimate_wn(f, pxx, fmin=5, fmax=np.inf):
    """Use frequencies between fmin and fmax to estimate the white noise level."""
    return np.sqrt(np.median(pxx[((f > fmin) & (f < fmax))]))


def bin_psd(f, pxx):
    """Bin a periodogram in log space."""
    bins = np.logspace(np.log10(f[0]), np.log10(f[-1]) + 1e-8, endpoint=True, num=25)
    digitized = np.digitize(f, bins)
    with np.errstate(invalid='ignore'):
        binned_pxx = np.bincount(digitized, weights=pxx) / np.bincount(digitized)
        binned_f = np.bincount(digitized, weights=f) / np.bincount(digitized)
    # do not use empty bins
    return binned_f[~np.isnan(binned_f)], binned_pxx[~np.isnan(binned_pxx)]


def fit_model(f, pxx, *, bin: bool):
    """Fit a 1/f model to a PSD."""
    # estimate white noise level before eventual binning
    p0 = [
        estimate_wn(f, pxx),  # sigma
        1,  # alpha
        0.1 * f[-1],  # fknee
        0.01 * f[-1],  # f0
    ]
    bounds = (
        # avoid singularities by using machine precision for lower bounds
        [0, 0.1, np.finfo(f.dtype).eps, np.finfo(f.dtype).eps],
        [np.inf, 10, f[-1], f[-1]],
    )

    if bin:
        f, pxx = bin_psd(f, pxx)

    return curve_fit(
        log_psd_model,
        f,
        np.log10(pxx),
        p0=p0,
        bounds=bounds,
        nan_policy='raise',
    )


# ____________________________________________________________
# PSD estimation


def fit_psd_to_tod(
    tod,
    fsamp,
    detrend: str = 'linear',
    welch: bool = True,
    nperseg: int = 8192,
    bin: bool = True,
):
    if welch:
        f, psd = scipy.signal.welch(tod, fs=fsamp, detrend=detrend, nperseg=nperseg)
    else:
        f, psd = scipy.signal.periodogram(tod, fs=fsamp, detrend=detrend, window='hann')

    return fit_model(f[1:], psd[1:], bin=bin)


def log_psd_model(x, sigma, alpha, fk, f0):
    return 2 * np.log10(sigma) + np.log10(1 + ((x + f0) / fk) ** -alpha)


def psd_model(x, sigma, alpha, fk, f0):
    return sigma**2 * (1 + ((x + f0) / fk) ** -alpha)


def psd_to_invntt(psd: npt.NDArray, correlation_length: int) -> npt.NDArray:
    """Compute the inverse autocorrelation function from PSD values
    The result is apodized and cut at the specified correlation length.
    """
    invntt = np.asarray(np.fft.irfft(1 / psd))[..., :correlation_length]
    return apodize(invntt)


def psd_to_ntt(psd: npt.NDArray, correlation_length: int) -> npt.NDArray:
    """Compute the autocorrelation function from PSD values.
    The result is apodized and cut at the specified correlation length.
    """
    ntt = np.asarray(np.fft.irfft(psd))[..., :correlation_length]
    return apodize(ntt)


def apodize(a):
    window = apodization_window(a.shape[-1])
    return a * window


def apodization_window(size: int, kind: str = 'chebwin') -> npt.NDArray:
    if kind == 'gaussian':
        q_apo = 3  # apodization factor: cut happens at q_apo * sigma in the Gaussian window
        window_type = ('general_gaussian', 1, 1 / q_apo * size)
    elif kind == 'chebwin':
        at = 150  # attenuation level (dB)
        window_type = ('chebwin', at)
    else:
        msg = f'Apodization window {kind!r} is not supported.'
        raise RuntimeError(msg)

    window = np.array(get_window(window_type, 2 * size))
    return np.fft.ifftshift(window)[:size]


def folded_psd(inv_n_tt: npt.NDArray, fft_size: int) -> npt.NDArray:
    """Returns the folded Power Spectral Density of a one-dimensional vector.

    Args:
        inv_n_tt: The inverse autocorrelation function of the vector.
        fft_size: The size of the FFT to use (at least twice the size of ``inv_n_tt``).
    """
    kernel = _get_kernel(inv_n_tt, fft_size)
    psd = 1 / np.abs(np.fft.rfft(kernel, n=fft_size))
    # zero out DC value
    # psd[0] = 0
    return psd


def _get_kernel(n_tt: npt.NDArray, size: int) -> npt.NDArray:
    lagmax = n_tt.size - 1
    padding_size = size - (2 * lagmax + 1)
    if padding_size < 0:
        msg = f'The maximum lag ({lagmax}) is too large for the required kernel size ({size}).'
        raise ValueError(msg)
    return np.concatenate((n_tt, np.zeros(padding_size), n_tt[-1:0:-1]))


def effective_ntt(invntt: npt.NDArray, fft_size: int) -> npt.NDArray:
    func = np.vectorize(folded_psd, signature='(m),()->(n)')
    effective_psd = func(invntt, fft_size)
    lagmax = invntt.shape[-1]
    return psd_to_ntt(effective_psd, lagmax)


def autocorr_to_psd(autocorr, fft_size: int):
    kernel = _get_kernel(autocorr, fft_size)
    psd = np.abs(np.fft.rfft(kernel, n=fft_size))
    # psd[0] = 0
    return psd


# ____________________________________________________________
# TOD manipulation


def sim_noise(
    samples=0,
    realization=0,
    detindx=0,
    sindx=0,
    telescope=0,
    fsamp=1.0,
    py=False,
    autocorr=None,
    freq=None,
    psd=None,
    use_toast=False,
    verbose=False,
):
    """
    Simulates a noise timestream according to a given autocorrelation function / PSD.

    Args:
        samples (int, optional): Number of samples to generate
        realization (int, optional): RNG realization
        detindx (int, optional): Detector index
        sindx (int, optional): Session index
        telescope (int, optional): Telescope ID
        fsamp (float, optional): Sample rate
        py (bool, optional): Whether to use the Python implementation instead of C. Defaults to False.
        autocorr (NDArray, optional): The autocorrelation function. Is needed to use the C routine.
        freq (NDArray, optional): Array containing the PSD frequencies. Is not needed if `autocorr` is given.
        psd (NDArray, optional): Array containing the PSD values. Is not needed if `autocorr` is given. For use with the Python implementation, must contain exactly nfft // 2 + 1 values, where nfft is the next fast FFT size above `samples`.
        use_toast (bool, optional): Whether to use TOAST's timestream generation. If True, the user must provide `freq` and `psd`. Defaults to False.

    Returns:
        NDArray: The generated timestream
    """
    # Do not use TOAST if autocorr is given
    if autocorr is not None:
        use_toast = False

    if use_toast:
        if verbose:
            print('sim_noise: use TOAST routine')
        if freq is None:
            raise ValueError('freq must be provided when using TOAST routine')
        if psd is None:
            raise ValueError('psd must be provided when using TOAST routine')
        return sim_noise_timestream(
            realization=realization,
            telescope=telescope,
            sindx=sindx,
            detindx=detindx,
            rate=fsamp,
            samples=samples,
            freq=freq,
            psd=psd,
        )

    # Using C implementation (mappraiser)
    if not py:
        if autocorr is None:
            raise ValueError('autocorr must be provided if using C routine')
        if verbose:
            print('sim_noise: use C routine')
        tdata = np.zeros(samples)
        mappraiser.sim_noise_tod(
            samples,
            len(autocorr),
            autocorr,
            tdata,
            realization,
            detindx,
            sindx,
            telescope,
            fsamp,
        )
        return tdata

    # Using Python implementation (similar to TOAST but no resampling of PSD is done)
    if verbose:
        print('sim_noise: use Python code')
    fftlen = 2
    while fftlen <= samples:
        fftlen *= 2
    npsd = fftlen // 2 + 1

    norm = fsamp * float(npsd - 1)

    if autocorr is not None:
        # Compute the PSD
        lag = len(autocorr)
        circ_t = np.pad(autocorr, (0, fftlen - lag), 'constant')
        if lag > 1:
            circ_t[-lag + 1 :] = np.flip(autocorr[1:], 0)
        psd_final = np.abs(np.fft.rfft(circ_t, n=fftlen))
    else:
        # User wants to work directly with the PSD
        # We don't do any resampling, input PSD must be of the right size
        if psd is None:
            raise ValueError('psd must be provided if not using autocorr')
        # assert len(psd) == npsd
        psd_final = np.copy(psd)

    psd_final[0] = 0
    scale = np.sqrt(psd_final * norm)

    # gaussian Re/Im randoms, packed into a complex valued array

    key1 = int(realization) * int(4294967296) + int(telescope) * int(65536)
    key2 = int(sindx) * int(4294967296) + int(detindx)
    counter1 = 0
    counter2 = 0

    rngdata = rng.random(
        fftlen,
        sampler='gaussian',
        key=(key1, key2),
        counter=(counter1, counter2),
    ).array()

    fdata = np.zeros(npsd, dtype=np.complex128)

    # Set the DC and Nyquist frequency imaginary part to zero
    fdata[0] = rngdata[0] + 0.0j
    fdata[-1] = rngdata[npsd - 1] + 0.0j

    # Repack the other values.
    fdata[1:-1] = rngdata[1 : npsd - 1] + 1j * rngdata[-1 : npsd - 1 : -1]

    # scale by PSD
    fdata *= scale

    # inverse FFT
    tempdata = np.fft.irfft(fdata)

    # subtract the DC level- for just the samples that we are returning
    offset = (fftlen - samples) // 2

    DC = np.mean(tempdata[offset : offset + samples])
    tdata = tempdata[offset : offset + samples] - DC
    return tdata


def baseline(tod, valid, w0, rm=False, cp=False):
    """
    Computes a baseline (running average) for the given TOD.

    Args:
        tod (NDArray): The input timestream
        valid (NDArray): An array where positive values indicate valid samples
        w0 (int): The window of the running average has a width of 2*w0+1
        rm (bool, optional): Subtract the baseline from the original TOD. Defaults to False.
        cp (bool, optional): Return a copy of the timestream with the baseline removed, instead of removing it from the original one. Default to False.

    Returns:
        NDArray: The computed baseline
        (NDArray, NDarray): The computed baseline and copied+modified TOD
    """
    samples = len(tod)
    baseline = np.zeros(samples, dtype=float)
    if cp:
        work = np.copy(tod)
        mappraiser.remove_baseline(samples, work, baseline, valid, w0, True)
        return baseline, work

    mappraiser.remove_baseline(samples, tod, baseline, valid, w0, rm)
    return baseline


# ____________________________________________________________
# Gap-filling


def sim_constrained_block(
    init,
    finalize,
    w0,
    tod,
    tt,
    invtt,
    indices,
    realization=0,
    detindx=0,
    sindx=0,
    telescope=0,
    fsamp=1.0,
):
    """
    Computes a new TOD where gaps have been filled with a constrained realization of the noise.

    Args:
        init (_type_): _description_
        finalize (_type_): _description_
        w0 (_type_): _description_
        tod (_type_): _description_
        tt (_type_): _description_
        invtt (_type_): _description_
        indices (_type_): _description_
        realization (int, optional): _description_. Defaults to 0.
        detindx (int, optional): _description_. Defaults to 0.
        sindx (int, optional): _description_. Defaults to 0.
        telescope (int, optional): _description_. Defaults to 0.
        fsamp (float, optional): _description_. Defaults to 1.0.

    Returns:
        NDArray: A copy of the input TOD with gaps filled.
    """
    samples = len(tod)
    filled = np.copy(tod)
    lambd = len(tt)
    mappraiser.sim_constrained_block(
        init,
        finalize,
        samples,
        lambd,
        w0,
        tt,
        invtt,
        filled,
        np.copy(indices),
        realization,
        detindx,
        sindx,
        telescope,
        fsamp,
    )
    return filled
