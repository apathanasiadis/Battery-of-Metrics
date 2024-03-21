import jax
import jax.numpy as np
from jax._src.numpy.util import check_arraylike
import numpy
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from bomtemplate import utility
from tqdm import tqdm


def hilbert(x, N=None, axis=-1):
    check_arraylike('hilbert', x)
    x = np.asarray(x)
    if x.ndim > 1:
        raise NotImplementedError("x must be 1D.")
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft.fft(x, N, axis=axis)
    if N % 2 == 0:
        h = np.zeros(N, Xf.dtype).at[0].set(1).at[1:N // 2].set(2).at[N // 2].set(1)
    else:
        h = np.zeros(N, Xf.dtype).at[0].set(1).at[1:(N+1) // 2].set(2)

    x = np.fft.ifft(Xf * h, axis=axis)
    return x


def FC_stream_phases(x):
    ah = np.angle(jax.vmap(hilbert)(x))
    dah = np.cos(jax.vmap(lambda a: a - a[:,None])(ah))
    return dah

class Zudah(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param
    
    def fit(self, X, y=None):
        if X.ndim == 3:
            self.n_participants, self.n_samples, self.n_nodes = X.shape      
        else:
            raise ValueError('ndim of input != 3.')

        # Return the transformer
        return self

    def transform(self, X):
        dah = jax.vmap(FC_stream_phases, in_axes=0)(X)
        numpy.testing.assert_(dah.ndim == 4)
        mask = np.ones((self.n_nodes, self.n_nodes))
        mask *= np.triu(mask, k=1)
        nonzero_idx = np.nonzero(mask)
        udah = dah[..., nonzero_idx[0], nonzero_idx[1]]
        zudah = jax.jit(jax.vmap(jax.vmap(utility.z_score, in_axes=0), in_axes=0))(udah)
        self.n_features = zudah.shape[-1]
        return zudah
    
class Flipping_Methods:
    
    def __init__(self, v1):
        self.v1 = v1
    
    def meanK(self):
        if np.mean(self.v1>0)>0.5:
            self.v1=-1*self.v1
        elif np.mean(self.v1>0)==.5 and np.sum(self.v1[(self.v1>0)])>-np.sum(self.v1[(self.v1<0)]):
            self.v1=-1*self.v1
        return self.v1
    
    def meanO(self):
        if np.mean(self.v1>0)>0.5:
            self.v1=-1*self.v1
        return self.v1
        

class V1s(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param
    
    def fit(self, X, y=None):
        '''following the standard sklearn procedute to get the estimator'''
        self.X = X
        if self.X.ndim == 3:
            self.n_participants, self.n_samples, self.n_features = self.X.shape      
        else:
            raise ValueError('ndim of input != 3.')
        # Return the transformer
        return self
    
    def transform(self, X):
        '''transform yields the v1s array'''
        dah = jax.vmap(FC_stream_phases, in_axes=0)(X)
        numpy.testing.assert_(dah.ndim == 4)
        _, evecs = np.linalg.eigh(dah)
        self.v1s = evecs[...,-1]
        return self.v1s
    
    def flip_v1s(self, option='markov'):
        '''
        flip v1s based on one of the given options:
        options = {'meanK', 'meanO', 'anchor', 'markov'}
        else it returns ValueError.
        Returns the flipped v1s.
        '''
        check_is_fitted(self, 'v1s')
        options = {'meanK', 'meanO', 'anchor', 'markov'}
        fm = Flipping_Methods()
        if option == 'meanK':
            # superslow -- kringelbach: https://github.com/decolab/pnas-neuromod/blob/master/LEiDA_PsiloData.m
            v1s_flipped = np.zeros_like(self.v1s)
            for p in tqdm(range(self.n_participants)):
                for s in range(self.n_samples):
                    v1 = self.v1s[p,s,:].copy()
                    v1 = ...
                    v1s_flipped = v1s_flipped.at[p,s,:].set(v1)
        elif option == 'meanO':
            # superslow -- olsen: https://github.com/anders-s-olsen/psilocybin_dynamic_FC/blob/main/pdfc_compute_eigenvectors.m
            v1s_flipped = np.zeros_like(self.v1s)
            for p in tqdm(range(self.n_participants)):
                for s in range(self.n_samples):
                    v1 = self.v1s[p,s,:].copy()
                    v1 = ...
                    v1s_flipped = v1s_flipped.at[p,s,:].set(v1)
        elif option == 'anchor':
            anchor = utility.get_anchor_node(self.X)
            v1s_flipped = np.where(self.v1s[...,anchor][...,None] > 0, -1*self.v1s, self.v1s)
        elif option == 'markov':
            anchor = utility.get_anchor_node(self.X)
            first_v1s = self.v1s[:,0,:].copy()
            first_v1s = np.where(first_v1s[:,anchor][:,None] > 0, -1*first_v1s, first_v1s)
            v1s_for_flip = np.concatenate((first_v1s[:,None,:], self.v1s[:,1:,:]), axis=1)
            # slow
            v1s_flipped = np.zeros_like(v1s_for_flip)
            for p in tqdm(range(self.n_participants)):
                v1 = np.array(v1s_for_flip[p,...].copy())
                for ti in range(1,self.n_samples):
                    s = np.sign(v1[ti-1,:] @ v1[ti,:])
                    v1 = v1.at[ti].set(v1[ti]*s)
                v1s_flipped = v1s_flipped.at[p,...].set(v1)
        else:
            raise ValueError(
                f"Invalid flipping option: {option}. Please choose from {options}.")
        return v1s_flipped



