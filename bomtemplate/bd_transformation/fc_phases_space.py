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

# Zudah ################
def FC_stream_phases(x):
    ah = np.angle(jax.vmap(hilbert)(x))
    dah = np.cos(jax.vmap(lambda a: a - a[:,None])(ah))
    return dah

class Zudah(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param'): # add option on which axis to z-score 
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
        # asserts whether by averaging out the last axis, the mean values for each sample (axis=1) is 0
        numpy.testing.assert_allclose(zudah.mean(-1),  0, atol=1e-5) 
        self.n_features = zudah.shape[-1]
        return zudah

# V1s ##################   
class Flipping_Methods: 
    def __init__(self, v1):
        self.v1 = v1 
    
    def meanK(self):
        v1_pos = self.v1 > 0
        mean_v1_pos = np.mean(v1_pos)
        sum_v1_pos = np.where(self.v1 > 0, self.v1, 0).sum()
        sum_v1_neg = np.where(self.v1 < 0, self.v1, 0).sum()
        condition = (mean_v1_pos > 0.5) | ((mean_v1_pos == 0.5) & (sum_v1_pos > -sum_v1_neg))
        return np.where(condition, -1*self.v1, self.v1)
    
    def meanO(self):
        return np.where(np.mean(self.v1>0)>0.5, -1*self.v1, self.v1)
    
    def anchor(self, anchor):
        return np.where(self.v1[anchor] > 0, -1*self.v1, self.v1)
    
    def markov(self, v_init, v1s): # used with jax.lax.scan
        dotprod = v_init @ v1s
        s = np.sign(dotprod)
        v1s *= s
        return v1s, v1s
        

class V1s(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param', return_evals = False):
        self.demo_param = demo_param
        self.return_evals = return_evals
    
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
        evals, evecs = np.linalg.eigh(dah)
        self.v1s = evecs[...,-1]
        if not self.return_evals:
            return self.v1s
        else:
            return self.v1s, evals[...,-1]
    
    def flip_v1s(self, option='markov'):
        '''
        flip v1s based on one of the given options:
        options = {'meanK', 'meanO', 'anchor', 'markov'}
        else it returns ValueError.
        Returns the flipped v1s.
        '''
        check_is_fitted(self, 'v1s')
        options = {'meanK', 'meanO', 'anchor', 'markov'}
        if option == 'meanK':
            # kringelbach: https://github.com/decolab/pnas-neuromod/blob/master/LEiDA_PsiloData.m
            wrapper = lambda x: Flipping_Methods(x).meanK()
            v1s_flipped = jax.vmap(jax.vmap(wrapper, in_axes=0), in_axes=0)(self.v1s)
        elif option == 'meanO':
            # olsen: https://github.com/anders-s-olsen/psilocybin_dynamic_FC/blob/main/pdfc_compute_eigenvectors.m
            wrapper = lambda x: Flipping_Methods(x).meanO()
            v1s_flipped = jax.vmap(jax.vmap(wrapper, in_axes=0), in_axes=0)(self.v1s)
        elif option == 'anchor':
            # flip vectors based on the sign of an anchor node derived from the data FC sum
            anchor = utility.get_anchor_node(self.X)
            wrapper = lambda x: Flipping_Methods(x).anchor(anchor)
            v1s_flipped = jax.vmap(jax.vmap(wrapper, in_axes=0), in_axes=0)(self.v1s)
        elif option == 'markov':
            # flip vectors sequentially making sure that the dotproduct of 2 successive vectors is always positive 
            anchor = utility.get_anchor_node(self.X)
            first_v1s = self.v1s[:,0,:].copy()
            first_v1s = np.where(first_v1s[:,anchor][:,None] > 0, -1*first_v1s, first_v1s)
            v1s_for_flip = np.concatenate((first_v1s[:,None,:], self.v1s[:,1:,:]), axis=1)
            fun = lambda a,b: Flipping_Methods(None).markov(a,b) # input for lax.scan
            v_init = np.ones(self.v1s.shape[-1])                 # initialization for lax.scan
            wrapper = lambda v1s: jax.lax.scan(fun, v_init, v1s)[1]
            v1s_flipped = jax.jit(jax.vmap(wrapper))(v1s_for_flip)
        else:
            raise ValueError(
                f"Invalid flipping option: {option}. Please choose from {options}.")
        return v1s_flipped
    


#EiDA ############################
def get_cs(x, return_iPL = False):
    ah = np.angle(hilbert(x))
    cah = np.cos(ah)
    sah = np.sin(ah)
    if return_iPL:
        pcah = cah[:,None] @ cah[None,:]
        psah = sah[:,None] @ sah[None,:]
        iPL = pcah@pcah.T + psah@psah.T
        return ah, cah, sah, iPL
    else:
        return ah, cah, sah

def do_analytics(c, s):
    y = np.linalg.norm(c)
    o = np.linalg.norm(s)
    j = c.T @ s
    D = (y-o)**2 + 4*y**2
    sol = lambda D: ((y-o) + D)/(2*j)
    B1 = sol(np.sqrt(D))
    B2 = sol(-np.sqrt(D))
    # get v1, v2
    v1 = c+B1*s
    v2 = c+B2*s
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    # get l1, l2
    l1 = o + j/B1
    l2 = o + j/B2
    return (l1, v1), (l2, v2)  


class Eida(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param', return_iPL = False): # add option on which axis to z-score 
        self.demo_param = demo_param
        self.return_iPL = return_iPL
    
    def fit(self, X, y=None):
        if X.ndim == 3:
            self.n_participants, self.n_samples, self.n_nodes = X.shape      
        else:
            raise ValueError('ndim of input != 3.')

        # Return the transformer
        return self
    
    def transform(self, X):
        numpy.testing.assert_(X.ndim == 3)
        if self.return_iPL:
            self.ah, self.cah, self.sah, iPL = jax.vmap(jax.vmap(get_cs, in_axes=(0,None)), in_axes=(0,None))(X, self.return_iPL)
            (self.l1, self.v1), (self.l2, self.v2) =  jax.vmap(jax.vmap(do_analytics))(self.cah, self.sah)
            return np.stack((self.l1, self.l2)), np.stack((self.v1, self.v2)), iPL        
        else:
            self.ah, self.cah, self.sah = jax.vmap(jax.vmap(get_cs, in_axes=(0,None)), in_axes=(0,None))(X, self.return_iPL)
            (self.l1, self.v1), (self.l2, self.v2) =  jax.vmap(jax.vmap(do_analytics))(self.cah, self.sah)        
            return np.stack((self.l1, self.l2)), np.stack((self.v1, self.v2))
    
    def spectral_radius(self):
        return np.stack((self.l1.mean(axis=-1), self.l2.mean(axis=-1)))
    
    def spectral_metastability(self):
        return self.l1.std(axis=-1)
    
    def kuramoto_metastability(self):
        return np.abs(self.ah).mean(-1).std(-1)
    
    def reconfiguration_speed(self, iPL):
            '''input: iPL matrix (3d);
            returns: reconfiguration speed
            utilize its mean or std'''
            numpy.testing.assert_(iPL.ndim == 3)
            ut = np.triu_indices(self.n_nodes, k=1)
            iPL = iPL[:,ut[0], ut[1]]
            s = lambda iPL, jPL: 1 - np.abs(np.corrcoef(iPL, jPL))[1,0]
            reconf_speed = jax.vmap(s)(iPL[:-1], iPL[1:])
            return reconf_speed

