from os import mkdir
from os.path import exists, join
import itertools
from functools import partial

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import pearsonr

from tqdm import tqdm

from mne.connectivity import spectral_connectivity, phase_slope_index


def get_feature_build_func(method, verbose=None):

    if method == 'coh-alpha':
        f = partial(get_mne_spec_con_feats, sfreq=125., rhythm='alpha', method='coh')
    elif method == 'coh-beta':
        f = partial(get_mne_spec_con_feats, sfreq=125., rhythm='beta', method='coh')
    elif method == 'env-alpha':
        f = partial(get_envelope_feats, sfreq=125., rhythm='alpha')
    elif method == 'env-beta':
        f = partial(get_envelope_feats, sfreq=125., rhythm='beta')
    elif method == 'bands':
        f = partial(get_rhythm_feats, sfreq=125.)
    elif method == 'psi':
        f = partial(get_psi_feats, sfreq=125.)
    else:
        raise ValueError('Features method is not in allowed list')

    def wrapped(data_path, out_path):
        path_file_path = join(data_path, 'path_file.csv')
        path_df = pd.read_csv(path_file_path)
        # required columns check
        assert all([col in path_df.columns for col in ['fn', 'target']])

        features_dir_path = join(out_path, 'features')
        if not exists(features_dir_path):
            mkdir(features_dir_path)

        features_path = join(features_dir_path, method.replace('-', '_') + '.csv')

        new_rows = []


        for i, row in tqdm(path_df.iterrows(), total=len(path_df)):
            if verbose and i % 10 == 0:
                print('At file {}'.format(i + 1))
            try:
                path = join(data_path, row['fn'])
                df = pd.read_csv(path, index_col='time')
                new_row = f(df)
            except AssertionError:
                print('Error in file ' + row['fn'])
                continue
            except FileNotFoundError:
                print('Not found - ' + row['fn'])
                continue

            for col in path_df.columns:
                new_row[col] = row[col]
            new_rows.append(new_row)

        res_df = pd.DataFrame(new_rows)

        res_df.to_csv(features_path, index=False)

    return wrapped


def process_file(path, method, rhythm=None):

    if method == 'coherence':
        f = partial(get_mne_spec_con_feats, sfreq=125., rhythm=rhythm, method='coh')
    elif method == 'envelopes':
        f = partial(get_envelope_feats, sfreq=125., rhythm=rhythm)
    elif method == 'rhythms':
        f = partial(get_rhythm_feats, sfreq=125.)
    elif method == 'psi':
        f = partial(get_psi_feats, sfreq=125.)
    else:
        raise ValueError('Features method not in allowed list')

    df = pd.read_csv(path, index_col='time')
    d = f(df)

    return d


def get_alpha_filter(sfreq=500.):

    f_low_lb = 7
    f_low_ub = 8
    f_high_lb = 13
    f_high_ub = 14

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate

    freq = [0., f_low_lb, f_low_ub, f_high_lb, f_high_ub, nyq]
    gain = [0, 0, 1, 1, 0, 0]
    n = int(round(5 * sfreq)) + 1

    alpha_filter = signal.firwin2(n, freq, gain, nyq=nyq)

    return alpha_filter


def get_beta_filter(sfreq=500.):

    f_low_lb = 12
    f_low_ub = 13
    f_high_lb = 29
    f_high_ub = 30

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate

    freq = [0., f_low_lb, f_low_ub, f_high_lb, f_high_ub, nyq]
    gain = [0, 0, 1, 1, 0, 0]
    n = int(round(5 * sfreq)) + 1

    alpha_filter = signal.firwin2(n, freq, gain, nyq=nyq)

    return alpha_filter


def get_rhythm_feats(df, sfreq=500.):

    electrodes = df.columns

    rhythms_bounds = {
        'theta' : [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 45]
    }

    feats = {}

    for el in electrodes:
        freqs, psds = signal.welch(df[el], sfreq, nperseg=1024)
        psd_df = pd.DataFrame(data={'freqs': freqs, 'psds': psds})
        feats[el + '_alpha'] = psd_df.loc[
            (psd_df['freqs'] >= rhythms_bounds['alpha'][0]) &
            (psd_df['freqs'] <= rhythms_bounds['alpha'][1])]['psds'].sum()

        feats[el + '_beta'] = psd_df.loc[
            (psd_df['freqs'] >= rhythms_bounds['beta'][0]) &
            (psd_df['freqs'] <= rhythms_bounds['beta'][1])]['psds'].sum()

        feats[el + '_theta'] = psd_df.loc[
            (psd_df['freqs'] >= rhythms_bounds['theta'][0]) &
            (psd_df['freqs'] <= rhythms_bounds['theta'][1])]['psds'].sum()

        feats[el + '_gamma'] = psd_df.loc[
            (psd_df['freqs'] >= rhythms_bounds['gamma'][0]) &
            (psd_df['freqs'] <= rhythms_bounds['gamma'][1])]['psds'].sum()

    return feats


def get_mne_spec_con_feats(df, sfreq=500., rhythm='alpha', method='coh'):

    electrodes = df.columns

    res = spectral_connectivity(
        df[electrodes].values.T.reshape(1, len(electrodes), -1),
        method=method, sfreq=sfreq, verbose=False)

    coh_data = res[0]
    freqs = res[1]

    if rhythm == 'alpha':
        fmin = 8
        fmax = 13
    elif rhythm == 'beta':
        fmin = 13
        fmax = 30

    idx_start = np.where(freqs > fmin)[0][0]
    idx_end = np.where(freqs < fmax)[0][-1]

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        d['con_alpha_' + el_1 + '_' + el_2] = coh_data[idx_2, idx_1][idx_start:idx_end + 1].mean()

    return d


def get_envelope_feats(df, sfreq=500., rhythm='alpha'):

    electrodes = df.columns

    df = df.copy()
    if rhythm == 'alpha':
        filt = get_alpha_filter(sfreq=sfreq)
    elif rhythm == 'beta':
        filt = get_beta_filter(sfreq=sfreq)
    else:
        ValueError('Wrong rhythm specified')

    for el in electrodes:
        sig_alpha = np.convolve(filt, df[el], 'same')
        sig_analyt = hilbert(sig_alpha)
        sig_envelope = np.abs(sig_analyt)
        df[el + '_env'] = sig_envelope

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        series_1 = df[el_1 + '_env'].values[500:-500]
        series_2 = df[el_2 + '_env'].values[500:-500]
        d['env_cor_' + el_1 + '_' + el_2] = pearsonr(series_1, series_2)[0]

    return d


def get_psi_feats(df, sfreq=125.):

    electrodes = df.columns

    df = df.copy()
    alpha_filter = get_alpha_filter(sfreq=sfreq)

    df = df[electrodes]
    for el in electrodes:
        df[el] = np.convolve(alpha_filter, df[el], 'same')

    vals = df.values
    vals = vals.transpose(1, 0)
    vals = vals[None, :, :]

    psi, freqs, times, n_epochs, _ = phase_slope_index(vals, sfreq=sfreq, verbose=False)
    d = {}
    for i in range(psi.shape[0]):
        for j in range(i):
            d['psi_{}_{}'.format(electrodes[i], electrodes[j])] = psi[i, j, 0]
    return d
