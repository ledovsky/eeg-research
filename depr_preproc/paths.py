from os.path import join
from os import listdir

import pandas as pd


class PathDataFrameConstructor(object):
    def __init__(self):
        self.rows = []

    def process_path(self, path, is_patient, dataset_name):
        for fn in listdir(path):
            if fn[-3:] != 'raw':
                continue
            # ignore Solomentseva
            if 'solomentseva' in fn.lower():
                continue
            self.rows.append((join(path, fn), fn.replace('.raw', '.csv'), is_patient, dataset_name))

    def build_df(self):
        return pd.DataFrame(self.rows, columns=['full_path', 'fn', 'target', 'dataset'])


def get_path_df(raw_data_path):
    path_norma = join(raw_data_path, 'norma 3min/')
    path_norma_2 = join(raw_data_path, 'norma 8min/')
    path_norma_3 = join(raw_data_path, 'norma 3min cs/')
    path_patients = join(raw_data_path, 'patients 3min/')
    path_patients_2 = join(raw_data_path, 'patients 8min/')

    constructor = PathDataFrameConstructor()

    constructor.process_path(path_norma, 0, '3min')
    constructor.process_path(path_norma_2, 0, '8min')
    constructor.process_path(path_norma_3, 0, '3min_cs')
    constructor.process_path(path_patients, 1, '3min')
    constructor.process_path(path_patients_2, 1, '8min')

    return constructor.build_df()
