import mne


def create_epochs(raw):
    """
    Convert continuous EEG into epochs.
    """

    events, event_id = mne.events_from_annotations(raw)

    event_dict = {
        "Left Hand": event_id["769"],
        "Right Hand": event_id["770"],
        "Feet": event_id["771"],
        "Tongue": event_id["772"]
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True
    )

    # Keep only EEG channels
    epochs = epochs.pick("eeg")

    return epochs


def bandpass_filter(epochs):
    """
    Apply 8-30 Hz filter.
    """

    epochs.filter(l_freq=8, h_freq=30)

    return epochs