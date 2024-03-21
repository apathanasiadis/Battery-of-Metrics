import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt#, sosfiltfilt # fix this
import matplotlib.pyplot as plt
import pickle
import itertools
import math
import warnings

class Get_1000Brains_Data:

    def __init__(self, path_data):
        self.path_data = path_data
    
    def get_data(self, bins = False):
        '''returns output dict'''
        # Get data and whereabouts
        with open(os.path.join(self.path_data,'binned_data__subjs__ages_ses0.pickle'), 'rb') as f:
            binned_data__subjs__ages = pickle.load(f)
        Age_Intervals = np.load(self.path_data+'Age_Intervals.npy', allow_pickle=True)
        binned_data, binned_subj, binned_ages = binned_data__subjs__ages
        n_bins = len(binned_data)
        nan_id = [(1,3)] # known from previous analyses
        # delete the data arrays with nans from the nested af list of data
        for id in nan_id:
            del binned_data[id[0]][id[1]]
            del binned_ages[id[0]][id[1]]
            del binned_subj[id[0]][id[1]]
        self.BOLD_TR             = int(2.25e3) #msec #make sure thats correct because the assignment was confusing
        self.transient           = int(5e3/self.BOLD_TR)
        self.n_participants_per_bin = np.array([len(binned_data[i]) for i in range(n_bins)])
        self.n_participants = self.n_participants_per_bin.sum()
        self.n_samples, self.n_nodes = binned_data[0][0].shape
        if bins == False:
            ages_flat = np.asarray(list(itertools.chain.from_iterable(binned_ages)))
            data_3d = np.vstack([np.asarray(binned_data[i]) for i in range(n_bins)])
            output_dict = {
                'data_3d':data_3d,
                'ages':ages_flat,
                'TR': self.BOLD_TR,
                'transient': self.transient
            }
            return output_dict
        else:
            xlabels = []
            for _, j in enumerate(Age_Intervals):
                xlabels.append(f"{str(j.left)}-{str(j.right)}")
            output_dict = {
                'binned_data': binned_data,
                'binned_ages': binned_ages,
                'binned_subj': binned_subj,
                'TR': self.BOLD_TR,
                'transient': self.transient,
                'age_intervals': Age_Intervals,
                'xlabels':xlabels
            }
            return output_dict
    
    def split_train_test(self, train_percentage=0.8):
        '''returns train and test idx to be used over
          the participants' axis of data_3d or ages for example'''    
        train_idx = []
        test_idx = []
        j=0
        idx = np.arange(self.n_participants)
        for i,n in enumerate(self.n_participants_per_bin):
            train_idx.append(idx[j:j+int(train_percentage*n)])
            test_idx.append(idx[j+int(train_percentage*n):j+self.n_participants_per_bin[i]])
            j+=self.n_participants_per_bin[i]
        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
        return train_idx, test_idx
    
    def split_test_retest(self, removed_transient = True):
        '''returns `part` which is an integer to be used to split samples,
          i.e. over the samples axis for test->[:part] & for retest->[-part:]
        '''
        if removed_transient:
            nsamples = self.n_samples - self.transient
        else:
            nsamples = self.n_samples
        gap = math.ceil(60/self.BOLD_TR*1e3)
        part = (nsamples - gap)//2
        if np.round(part*self.BOLD_TR/1e3/60) < 5:
            warnings.warn("`part` is less than 4.5 minutes")
        return part
    



# filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, axis, lowcut, highcut, fs, fun=sosfilt, order=5):
    '''
    inputs: data, axis, lowcut, highcut, fs, fun=sosfilt, order=5
    returns filtered data
    fun: sosfilt, sosfiltfilt
    '''
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = fun(sos, data, axis)
    return y

def get_ps(data, axis):
    '''applies fft on the given data and gets the power spectrum (ps) out of it'''
    ft_data = np.fft.fft(data, axis=axis) # Compute the FFT
    ps = np.abs(ft_data)**2 # Compute the power spectrum
    return ps

def plot_filtered_ts(data, TR, amp_scaler=20):
    '''
    Inputs
    -------------------------------
    data: array
        data (after filtering - or not) of shape (samples, features)
    TR: float
        scanning rate in [ms]
    amp_scaler: float
        divider of the amplitude of the time-series for better visualization
    
    Outputs
    --------------------------------
    plot
    '''
    plt.figure(figsize=(15, 5))
    plt.plot(np.r_[:data.shape[0]]*TR*1e-3, # getting the actual scanning time -- shape*bold_tr*1e-3
            data/amp_scaler + np.r_[:data.shape[1]], # rescaling to get slighter overlaps
            'k',
            alpha=0.7)
    plt.grid(1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Region')
    plt.tight_layout();
