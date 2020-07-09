import numpy as np
import pandas as pd
from os.path import join
from os import listdir
import os
import mne
from mne.io import read_raw_edf
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tSNE
from scipy.spatial.distance import cdist
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from scipy.signal import hilbert
from scipy.signal import argrelextrema
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#dictionary with clustering algorithms
#c_alg = {'k_means' : get_clusters}

def get_filter(bounds = [8,13], sfreq=125.):
    """
    Args:
        bounds: np.array, (2,)
        sfreq: float
    Returns:
        filt: np.array, (int(round(1 * sfreq)) + 1,)
    """
    f_low_lb = bounds[0] - 1
    f_low_ub = bounds[0]
    f_high_lb = bounds[1]
    f_high_ub = bounds[1] + 1

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate

    freq = [0., f_low_lb, f_low_ub, f_high_lb, f_high_ub, nyq]
    gain = [0, 0, 1, 1, 0, 0]
    n = int(round(1 * sfreq)) + 1

    filt = signal.firwin2(n, freq, gain, nyq=nyq)

    return filt

def apply_filter(filt, sig):
    """
    Args: 
        filt: np.array, (int(round(1 * sfreq)) + 1,)
        sig: np.array, (n_points,)
    Returns:
        sig: np.array, (n_points,) 
    
    """

    sig = np.convolve(filt, sig, 'valid')
    #sig = hilbert(sig)
    #sig = np.abs(sig)
    return sig

def get_clusters(X, n=4, dist=cdist, dist_mode='cos', crit = 0.00001, alg_mode = None, swap = False):
    """
    Args: 
        X: np.array, (n_maximas, n_channels)
        n: int , number of microstates
        dist_mod: 'cos' | 'dist' | 'ort_dist'
        crit: float
        alg_mode: 'm_k-means' or  None
        swap: True or False
        
    Returns:
        centroids: np.array, (n, n_channels)
        labels: np.array, (n_maximas)
        code: float
        initial_random state: np.array, (n, n_channels)
        
    """
def get_clusters(X, n=4, dist=cdist, dist_mode='cos', crit = 0.00001, alg_mode = None, swap = False, starting_mode = None):
    x = X.shape[0]
    starting_points = np.random.randint(0,x,n)

    centroids = X[starting_points,:]
    if starting_mode == 'pca':
        pca = PCA(4)
        pca.fit_transform(X)
        centroids = pca.components_
    cent_history = []
    #cent_history.append(centroids)
    shift = np.ones(n)
    iter = 0
    while max(np.abs(shift))>crit and iter < 1000:
        iter +=1
        if dist_mode == 'dist':
            distances = dist(X, centroids)
            labels = distances.argmin(axis=1)
        elif dist_mode == 'ort_dist':
            norms=np.linalg.norm(centroids, axis=1)
            c = np.multiply(centroids,1/norms.reshape(norms.shape[0], 1))
            distances = np.zeros((X.shape[0],n))
            for i in range(X.shape[0]):
                for j in range(n):
                distances[i,j] = np.linalg.norm(np.dot(X[i], c[j])*c[j] - X[i])
            labels = distances.argmin(axis=1)
        elif dist_mode == 'cos':
            distances = np.zeros((X.shape[0],n))
            for i in range(X.shape[0]):
                for j in range(n):
                    distances[i,j] = np.abs(np.dot(X[i], centroids[j])/(np.linalg.norm(X[i])*np.linalg.norm(centroids[j])))#,np.abs(1-np.dot(X[i], centroids[j])/(np.linalg.norm(X[i])*np.linalg.norm(centroids[j]))))
            labels = distances.argmax(axis=1)
        shift = np.ones(n)
        if alg_mode == 'm_k-means':
            centroids = centroids.copy()
            for i in range(n):
                V = np.dot(X[labels == i, :].T, X[labels == i, :])
                pca = PCA(1)
                pca.fit_transform(V)
                e = centroids[i, :].copy() 
            centroids[i, :]= pca.components_.copy()/np.linalg.norm(pca.components_.copy())
            shift[i] = np.linalg.norm(e-centroids[i, :])

        else:
            centroids = centroids.copy()
            for i in range(n):
                e = centroids[i, :].copy() 
                centroids[i, :]= np.mean(X[labels == i, :], axis=0)
                shift[i] = np.linalg.norm(e-centroids[i, :])
        if swap == True and alg_mode == 'm_k-means':
            alg_mode = None
        elif swap == True and alg_mode == None:
            alg_mode = 'm_k-means'
        #print(max(shift))
        code = max(shift)
    return centroids, labels, code, X[starting_points,:]


def get_labels(X, cent_history, maximas, max_labels, dist, no_peak_points):
    """
    Args:
        X: np.array, (n_points x n_channels)
        no_peak_points: 'ffill' | 'centroids'
        cent_hist: list of length 1000, each element is np.array (n, n_channels)
        maximas: np.array (n_maximas,)
        max_labels: np.array (n_maximas,)
        dist: function to compute distance between two vectors
        n: int
    Returns:
        labels: np.array (n_points)
    """
    if no_peak_points =='centroids':
        distances = dist(X, cent_history)
        labels = distances.argmin(axis=1)
    labels = np.zeros(X.shape[0])
    
    if no_peak_points=='ffill':
        i = 0
        for j in range(X.shape[0]):
            if i != len(maximas)-1:
                if abs(j-maximas[i])<abs(j-maximas[i+1]):
                    labels[j] = max_labels[i]
                else:
                    i=i+1
                    labels[j] = max_labels[i]
            else:
                labels[j] = max_labels[i]

    
    return labels

def cut_low_gfp(X, gfp, ratio=0.15):
    """
    Args:
        X: np.array, (n_poins, n_channels)
        gfp: np.array, (n_points)
        ratio: float
    Returns:
        X: np.array, (int(ratio*n_points), n_channels)
    
    """
    N = int(ratio*len(gfp))
    #print(len(ind[:N]))
    ind = sorted(range(len(gfp)), key=lambda i:gfp[i])
    X = X[ind[:N]]
    return X

def assign_labels_auto(centroids_0, centroids, labels_0, n=4):
  
    """
    Args:
        centroids_0: centroids of new patient, np.array, (n, n_channels)
        centroids: centroids of old patients, np.array, (n, n_channels)
        labels_0: np.array, (n_points)
    Returns:
        labels_0: np.array, (n_points)
        labels: np.array, (n)
        distances: np.array, (n,n)

    """
  
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distances[i,j] = np.abs(np.dot(centroids_0[i], centroids[j]))/(np.linalg.norm(centroids_0[i])*np.linalg.norm(centroids[j]))
    labels = distances.argmax(axis=1)
    for i,j in enumerate(labels_0):
        labels_0[i]=labels[labels_0[i]]
    return labels_0, labels, distances

def calc_microstates(data, window, no_peak_points='ffill',bounds = [8,13], sfreq=125., clustering_alg='k_means', normalize=False, n=4, dist=cdist):
    """
    Args:
        data: np.array, (n_points x n_channels)
        no_peak_points: 'ffill' | 'centroids'
        window: int
        bounds: np.array, (2,)
        clustering_alg: name of clustering algorithm
        normalize: True | False
        dist: function to compute distance between two vectors
        n: int
    Returns:
        clusters: np.array (n_points)
        centroids: np.array (n_clusters x n_channels)
        gfp: np.array (n_points)
        labels: np.array (n_points)
        max_labels: np.array (n_maximas)

    """
    
    filt = get_filter(bounds, sfreq)
    filtered = np.linspace(0,(data.shape[0]-int(sfreq))*8, (data.shape[0]-int(sfreq))*8)
    for i in df.columns[1:]:
        a = apply_filter(filt, df[i])
        filtered = np.vstack((filtered,a))
    X = filtered[1:].T
    gfp = np.var(X, axis=1)
    maximas = argrelextrema(gfp, np.greater, order=window)
    maximal_states = X[maximas[0],:]
    if normalize == True:
        norms=np.linalg.norm(X, axis=1)
        X = np.multiply(X,1/norms.reshape(norms.shape[0], 1))
    history, max_labels, _, _ =  get_clusters(X,n,crit = 1e-7,dist_mode = 'ort_dist', alg_mode = 'm_k-means')
    labels = get_labels(X, history, maximas[0], max_labels, dist, no_peak_points)
    return clusters, history, gfp, labels, max_labels

def cluster_patients(dir,n=4)
    
    """

    Args:
        dir: string, Name of directory with csv files
    """
    label_list = []
    gfp_list = []
    max_labels_list = []
    iter = 0
    for i in os.listdir(dir):
        if i != 'path_file.csv':
            iter+=1
            df = pd.read_csv(dir+i)

            filtered = get_filtered_data(df)

            X = filtered[1:].T
    
            #norms=np.linalg.norm(X, axis=1)
            #X = np.multiply(X,1/norms.reshape(norms.shape[0], 1))
            gfp, maximas, maximal_states = compute_gfp_maximas(X, 7)
            #[maximas[0],:]
            #X = cut_low_gfp(X, gfp, ratio=0.05)
            history, max_labels, _, init = get_clusters(X,n,crit = 1e-7,dist_mode = 'ort_dist', alg_mode = 'm_k-means')#, starting_mode = 'pca')
    
            labels = get_labels(X,history)
            if iter == 1:
                #X_old = history
                stack = history
            else:
                #print(np.max(X_old - history))
                stack = np.vstack((stack,history))
            #X_old = history
            re_labels, l, d = assign_labels_auto(history, prev, labels, n)
            prev = history.copy()
    
    #if not (0 in l and 1 in l and 2 in l and 3 in l):
     # print('bum')
      #break

    total_centroids, _, _, _  = get_clusters(stack,crit = 1e-7,dist_mode = 'cos')

    return total_centroids

def visualise_clusters_3d(X,labels,embedding='tSNE',n=4):
    """
    Args:
        X: np.array, (n_points, n_channels)
        labels: np,array (n_points,)
        embedding: 'tSNE' | 'PCA'
    """
    if embedding = 'tSNE':
        emb = tSNE(3)
    elif embedding = 'PCA':
        emb = PCA(3)
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    X = embedding.fit_transform(X)

    ax.scatter(
      xs=X[:,0], 
      ys=X[:,1], 
      zs=X[:,2], 
      c=labels)
    plt.show()

def visualise_topomap(v):
    
    """
    Args: 
        v: np.array, (n_channels)

    """

    channels_to_use = [
    't6',
    't4',
    'o1',
    'f8',
    'p4',
    'c4',
    't3',
    'f7',
    'f3',
    'o2',
    'f4',
    'c3',
    'p3',
    't5',
    'cz',
    'fp1',
    'fp2',
    'pz',
    'fz'
    ]
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    # create Info object to store info
    info = mne.io.meas_info.create_info(ten_twenty_montage.ch_names, sfreq=256, ch_types="eeg")
    #using temporary RawArray to apply mongage to info
    mne.io.RawArray(np.zeros((len(ten_twenty_montage.ch_names), 1)), info, copy=None).set_montage(ten_twenty_montage)

    # pick channels
    channels_to_use_ = [ch for ch in info.ch_names if ch.lower() in channels_to_use]
    info.pick_channels(channels_to_use_);

    # get positions
    _, pos, _, ch_names, _, _, _ = mne.viz.topomap._prepare_topomap_plot(info, 'eeg');

    mne.viz.plot_topomap(v, pos, names=ch_names, show_names=True)

def compute_transition_matrix(mode, labels, max_labels, n=4):
    """
    Args:
        labels: np.array, (n_points)
        mode: 'state_to_state' | 'stepwise'
    Returns:
        T: np.array, (n,n)
    """
    
    T = np.zeros((n,n))
    norms = np.zeros(n)
    if mode == 'state_to_state':
        labels = max_labels
    for i,j in enumerate(labels[1:]):
        p=i+1
        T[labels[p-1],labels[p]]=T[labels[p-1],labels[p]]+1
        norms[labels[p-1]]= norms[labels[p-1]]+1
    T = np.dot(np.diag(1/norms),T)
    
    return T

def compute_distribution(labels,n=4):
    """
    Args:
        labels: np.array, (n_points)
    Returns:
        c: np.array, (n)
    """
    c=np.zeros(n)
    for i in labels:
        c[i]=c[i]+1
    c = c/len(labels)
    return c

def compute_time_distribution(labels,n=4):
        
    """
    Args:
        labels: np.array, (n_points)
    Returns:
        time_dist: list of length n, each element is list of length n_points
    """
    
    
    time_dist = [[] for _ in range(n) ]
    flag = 0
    prev = labels[0]
    counter = 0

    for i in labels:
        if prev != i:
            time_dist[prev].append(counter)
            counter = 0
            prev = i
        else:
            counter = counter + 1
    for i in range(n):
        time_dist[i]=[j/len(labels) for j in time_dist[i]]
  return time_dist

def spatial_corr(V,u):
    """

    Args: 
        V: np.array, (n, n_channels)
        u: np.array, (n_channels)
    Returns:
        G: np.array, (n)
    """
    
    s = V.shape[0]
    G = np.zeros(s)
    for i in range(s):
        C=np.corrcoef(V[i],u)
        G[i] = C[0,1]
    return G

def GEV(labels, gfp, X, cent_history,n=4):

    """
    Args:
        labels: np.array, (n_points)
        gfp: np.array, (n_points)
        X: np.array, (n_points, n_channels)
        cent_history: np.array, (n, n_channels)
    
    Returns:
        np.array, (n)
    """
    C = np.zeros((n, len(gfp)))
    for i in range(n):
        C[i] = spatial_corr(X,cent_history[i])
    return np.dot(C**2,gfp**2)/np.sum(gfp**2)


def mutual(labels,lag,n=4):
    
    """
    Args:
        labels: np.array, (n_points)
        lag: int
    Returns:
        MI: float
    """
    
    
    paired = np.zeros((n,n))
    norm = len(labels[:-lag])

    for i in range(n):
        for j,k in enumerate(labels[:-lag]):
        paired[k,labels[j+lag]]=paired[k,labels[j+lag]]+1
    paired = paired / norm
    dist = compute_distribution(labels)
    o = np.outer(dist,dist)
    #print(paired)
    s = paired/o
    #print(s)
    MI = np.sum(np.log(s)*paired)

  return MI


def get_microstate_features(lables, max_labels, X, gfp, history,  sfreq=125., n=4, mode = 'state_to_state'):
    """
    Args:
        mode = 'state_to_state' | 'stepwise"
        labels: np.array, (n_points)
        max_labels: np.array, (n_maximas)
        X: np.array, (n_points, n_channels)
        gfp: np.array, (n_points)
    
    Returns: 
        feature_dict: dictionary
    """

    
    T = compute_transition_matrix(mode, labels, max_labels, n)
    distribution = compute_distribution(labels)
    time_dist = compute_time_distribution(labels)
    
    t_mean = np.zeros(n)
    t_median = np.zeros(n)
    t_max = np.zeros(n)
    t_variance = np.zeros(n)
    for i in range(n):
        state_time_dist = pd.Series(time_dist[i])
        t_mean[i] = state_time_dist.mean()
        t_median[i] = state_time_dist.median()
        t_max[i] = state_time_dist.max()
        t_variance[i] = state_time_dist.variance()
    
    MI = []
    for i in range(1,1000):
        MI.append(mutual(labels,i))    
    MI = np.array(MI)
    
    gev = GEV(labels, gfp, X, history, n):
    pca = PCA(1)
    X_1 = pca.fit_transform(X)
    
    feature_dict = {'Transition matrix' : T,  time_statistics = [t_mean, t_median, t_max, t_variance], 'MI' : MI, 'GEV' : GEV, 'First component' : X_1}
    return feature_dict


