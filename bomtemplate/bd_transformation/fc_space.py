from functools import partial
import jax
import jax.numpy as np
import numpy
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from bomtemplate import utility
from tqdm import tqdm


def get_FC_stream(ts, nn, TR, window_length):
    windowed_data = numpy.lib.stride_tricks.sliding_window_view(
        ts, (int(window_length/TR), nn), axis=(0, 1)).squeeze()
    n_windows = windowed_data.shape[0]
    fc_stream = np.asarray(
        [np.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)])
    return fc_stream

@partial(jax.jit, static_argnums=(1, 2, 3))
def get_FCD_ut(ut_fc_stream, TR, window_length, return_fcd = False):
    '''get collection of ut(FCDs) given a collection of ut(FC_stream)'''
    fcd = np.corrcoef(ut_fc_stream, rowvar=True)
    ut_idx = np.triu_indices_from(fcd, k=int(window_length/TR))
    fcd_ut = fcd[ut_idx]
    if return_fcd:
        return fcd_ut, fcd
    else:
        return fcd_ut

class FCD_measures(TransformerMixin, BaseEstimator):
    def __init__(self, demo_param='demo_param', TR=1.0, window_length=30.0, option='fcd', masks={}, positive=True, return_fcd=False):
        self.demo_param = demo_param
        self.TR = TR
        self.window_length = window_length
        self.option = option
        self.masks = masks
        self.positive = positive
        self.return_fcd = return_fcd

    def fit(self, X, TR=1.0, window_length=30):
        '''fitting for a transfomation'''
        if X.ndim == 3:
            self.n_participants, self.n_samples, self.n_features = X.shape      
        else:
            raise ValueError('ndim of input != 3.')
        self.TR = TR
        self.window_length = window_length
        
        return self
    
    def transform(self, X):
        '''transform to a collection of ut(fcds) or ut(fc_streams)
        for each of the given masks
        '''
        check_is_fitted(self, 'n_features')
        # fc_stream = jax.vmap(get_FC_stream, in_axes = (0, None, None, None))(X, self.n_features, self.TR, self.window_length)  # doesnt work 
        fc_stream = []
        for i in range(self.n_participants):  # can get optimized.
            fc_stream.append(get_FC_stream(X[i], self.n_features, self.TR, self.window_length))
        fc_stream = np.asarray(fc_stream)
        if self.positive == True:
            # mask out the negative correlations by multiplying with 1 the positive 
            # and with 0 the negative correlations
            fc_stream *= fc_stream > 0
        options = {'fc_stream', 'fcd'}
        self.mask_full = np.ones((self.n_features, self.n_features))
        FC_streams = []
        FCDs = []
        if len(self.masks) == 0:
            self.masks["full"] = self.mask_full
        for key in self.masks.keys():
            mask = self.masks[key]
            # get the upper triangle of the given mask
            mask *= np.triu(self.mask_full, k=1)
            nonzero_idx = np.nonzero(mask)
            self.ut_fc_stream = fc_stream[..., nonzero_idx[0], nonzero_idx[1]]
            if self.option == 'fc_stream':
                del FCDs
                return self.ut_fc_stream
            elif self.option == 'fcd':
                fcd = jax.vmap(get_FCD_ut, in_axes=(0, None, None, None))(self.ut_fc_stream, self.TR, self.window_length, self.return_fcd)
                del FC_streams
                return fcd
            else:
                raise ValueError(
                    f"Invalid flipping option: {self.option}. Please choose from {options}.")


# class for edge fc
            