import jax
import jax.numpy as np
import numpy
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from bomtemplate import utility
from tqdm import tqdm


def get_FC_stream(ts, nn, TR, window_length):
    windowed_data = np.lib.stride_tricks.sliding_window_view(
        ts, (int(window_length/TR), nn), axis=(0, 1)).squeeze()
    n_windows = windowed_data.shape[0]
    fc_stream = np.asarray(
        [np.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)])
    return fc_stream

def get_FCD_ut(TR, window_length, ut_fc_stream):
    '''get collection of ut(FCDs) given a collection of ut(FC_stream)'''
    fcd = np.corrcoef(ut_fc_stream, rowvar=True)
    ut_idx = np.triu_indices_from(fcd, k=int(window_length/TR))
    fcd_ut = fcd[ut_idx]
    return fcd_ut

class FCD_measures(TransformerMixin, BaseEstimator):
    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param
        self.TR = TR
        self.window_length = window_length

    def fit(self, X, TR=1.0, window_length=30):
        '''fitting for a transfomation'''
        if X.ndim == 3:
            self.n_participants, self.n_samples, self.n_features = X.shape      
        else:
            raise ValueError('ndim of input != 3.')
        self.TR = TR
        self.window_length = window_length
        
        return self
    
    def transform(self, X, option='fcd', masks=None, positive=True):
        '''transform to a collection of ut(fcds) or ut(fc_streams)
        for each of the given masks
        '''
        check_is_fitted(self, 'n_features')
        self.fc_stream = jax.jit(jax.vmap(get_FC_stream, in_axes = 0))(X, self.n_features, self.TR, self.window_length)
        if positive == True:
            # mask out the negative correlations by multiplying with 1 the positive 
            # and with 0 the negative correlations
            self.fc_stream *= self.fc_stream > 0
        options = {'fc_stream', 'fcd'}
        self.mask_full = np.ones((self.n_features, self.n_features))
        FC_streams = []
        FCDs = []
        if len(masks) == 0 or masks==None:
            masks["full"] = self.mask_full
        for key in masks.keys():
            mask = masks[key]
            # get the upper triangle of the given mask
            mask *= np.triu(self.mask_full, k=1)
            nonzero_idx = np.nonzero(mask)
            fc_stream_masked = self.fc_stream[..., nonzero_idx[0], nonzero_idx[1]]
            if option == 'fc_stream':
                FC_streams.append(fc_stream_masked)
                del FCDs
                return FC_streams
            elif option == 'fcd':
                fcd = jax.jit(jax.vmap(get_FCD_ut, in_axes=0))(fc_stream_masked, self.TR, self.window_length)
                FCDs.append(fcd)
                del FC_streams
                return FCDs
            else:
                raise ValueError(
                    f"Invalid flipping option: {option}. Please choose from {options}.")


# class for edge fc
            