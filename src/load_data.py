import mne


def load_subject(file_path):
    """
    Load one subject's EEG recording.
    """

    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Correct channel types
    raw.set_channel_types({
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    })

    return raw