import os
import mne

def load_subject(subject_id):
    file_path = os.path.join(
        "datasets",
        f"A{subject_id:02d}T.gdf"
    )

    print("Loading:", file_path)

    raw = mne.io.read_raw_gdf(file_path, preload=True)

    raw.set_channel_types({
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    })

    return raw