import numpy as np


def calc_microstates(data, no_peak_points='ffill', sfreq=125.):
    """
    Args:
        data: np.array, (n_points x n_channels)
        no_peak_points: 'ffill' | 'centroids'
    Returns:
        clusters: np.array (n_points)
        centroids: np.array (n_clusters x n_channels)
    """
    return clusters, centroids


def assign_labels(clusters, centroids, tagged_samples):
    return labels


def get_microstate_features(lables, sfreq=125.):
    return feature_dict
