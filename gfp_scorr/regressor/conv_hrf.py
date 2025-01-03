import numpy as np

def conv_hrf(correlation_series, tr, hrf=None):
    """
    Convolve the correlation time series with a hemodynamic response function (HRF).

    Parameters:
        correlation_series: ndarray
            Time series of correlation values.
        tr: float
            Repetition time (TR) of the fMRI acquisition.
        hrf: ndarray, optional
            Predefined HRF to convolve with. If None, a canonical HRF will be used.

    Returns:
        regressor: ndarray
            Convolved time series to be used as an fMRI regressor.
    """
    if hrf is None:
        from nilearn.glm.first_level import spm_hrf
        hrf = spm_hrf(tr)

    regressor = np.convolve(correlation_series, hrf)[:len(correlation_series)]
    return regressor