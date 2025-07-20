import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from scipy.signal import get_window

WELCH_SEGMENT_DURATION = 300  # 5 minutes


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
        log_model,
        f,
        np.log10(pxx),
        p0=p0,
        bounds=bounds,
        nan_policy='raise',
    )


def log_model(x, sigma, alpha, fk, f0):
    return 2 * np.log10(sigma) + np.log10(1 + ((x + f0) / fk) ** -alpha)


def model(x, sigma, alpha, fk, f0):
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
