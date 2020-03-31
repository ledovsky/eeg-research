from os import listdir
from os.path import join
import re

import numpy as np
import pandas as pd
from mne.io import read_raw_edf

to_exclude = [
    'partisipant17_TBI_ZG_ok.EDF'
]

class PathDataFrameConstructor(object):
    def __init__(self):
        self.rows = []

    def process_path(self, path, target_value, dataset_name, ext='edf'):
        for fn in listdir(path):
            print(fn)
            if fn in to_exclude:
                continue

            if fn[-3:].lower() != ext:
                continue

            full_path = join(path, fn)
            print(full_path)

            sample = read_raw_edf(full_path, stim_channel=False, exclude='Status')

            d = {}
            d['full_path'] = full_path
            d['fn'] = fn.lower().replace(ext, 'csv')
            d['target'] = target_value
            d['dataset_name'] = dataset_name

            d['sfreq'] = sample.info['sfreq']
            d['channels'] = '|'.join(sorted(sample.info['ch_names']))
            d['n_channels'] = len(sample.info['ch_names'])
            # found = re.findall('\d+', fn)[0]
            # if found:
            #     d['age'] = found
            # else:
            #     d['age'] = np.nan

            eeg_channels = [ch for ch in sample.info['ch_names'] if 'EEG' in ch]

            d['channels_eeg'] = '|'.join(sorted(eeg_channels))
            d['n_channels_eeg'] = len(eeg_channels)

            d['seconds'] = sample.get_data().shape[1] / d['sfreq']

            self.rows.append(d)

    def build_df(self):
        return pd.DataFrame(self.rows)


def get_path_df(raw_data_path):
    path_chmt = join(raw_data_path, 'sredne-tyazhelaya_CHMT')
    path_norm = join(raw_data_path, 'vzrosl_clean')

    constructor = PathDataFrameConstructor()

    constructor.process_path(path_chmt, 'chmt', 'chmt')
    constructor.process_path(path_norm, 'norm', 'norm')

    return constructor.build_df()

