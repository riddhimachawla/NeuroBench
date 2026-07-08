import mne


def bandpass_filter(raw, low=8, high=30):

    raw = raw.copy()

    raw.filter(
        l_freq=low,
        h_freq=high,
        verbose=False
    )

    return raw


def notch_filter(raw, freq=50):

    raw = raw.copy()

    raw.notch_filter(
        freq,
        verbose=False
    )

    return raw