from scipy.signal import butter, filtfilt

def preprocess_eeg(eeg_data, sfreq, low=1, high=30):
    """
    Band-pass filter the EEG data.

    Parameters:
        eeg_data: ndarray
            EEG data array with shape (n_channels, n_samples).
        sfreq: float
            Sampling frequency of the EEG data.
        low: float
            Low cutoff frequency for the filter.
        high: float
            High cutoff frequency for the filter.

    Returns:
        eeg_filtered: ndarray
            Band-pass filtered EEG data.
    """
    b, a = butter(2, [low / (sfreq / 2), high / (sfreq / 2)], btype='band')
    eeg_filtered = filtfilt(b, a, eeg_data, axis=1)
    return eeg_filtered