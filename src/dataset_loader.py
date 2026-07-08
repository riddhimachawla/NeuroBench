import os
import mne
import numpy as np

EVENT_ID = {
    "left_hand": 7,
    "right_hand": 8,
    "feet": 9,
    "tongue": 10
}


class EEGDataset:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_raw(self):

        raw = mne.io.read_raw_gdf(
            self.filepath,
            preload=True
        )

        raw.rename_channels(
            lambda x: x.strip(".")
        )

        return raw

    def get_events(self, raw):

        events, _ = mne.events_from_annotations(raw)

        return events

    def create_epochs(
        self,
        raw,
        events,
        tmin=2,
        tmax=6
    ):

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            eog=False
        )

        epochs = mne.Epochs(
            raw,
            events,
            event_id=EVENT_ID,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            preload=True,
            baseline=None
        )

        X = epochs.get_data()

        y = epochs.events[:, -1]

        return X, y