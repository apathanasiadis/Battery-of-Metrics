import jax
import jax.numpy as np
import numpy
import scipy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm


def labels2occup(labels, min_len):
    bc = np.bincount(labels, length=min_len)
    occup = bc / bc.sum()
    return occup

cossim = lambda a, b: np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

test_statistic = lambda d1, d2: d1.mean() - d2.mean()

class Fingerprinting(BaseEstimator):

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        '''X, y follow the standard sklearn shapes
        X: train labels
        y: test labels
        '''
        X = check_array(X)
        y = check_array(y)
        self.n_participants_ = X.shape[0]
        self.n_states = unique_labels(X.ravel()).size
        return self
    
    def transform(self, X, y, option = 'occurence ratio'):
        options = {'occurence ratio'}
        check_is_fitted(self, 'n_states')
        if option == 'occurence ratio':
            self.X_ = jax.vmap(labels2occup, in_axes=(0, None))(X, self.n_states)
            self.y_ = jax.vmap(labels2occup, in_axes=(0, None))(y, self.n_states)
        else:
            raise ValueError(
                f"Invalid flipping option: {option}. Please choose from {options}.")
        return self.X_, self.y_

    def score_intrasim_vs_intrasim(self, num_permutations):
        '''returns distribution of permuted_statistics, observed_statistic, p_value'''
        self.intrasim = np.array([ cossim(self.X_[p], self.y_[p]) for p in range(self.n_participants_) ])
        self.intersim = np.array([ cossim(self.X_[p], self.y_[ numpy.random.randint(self.n_participants_) ]) for p in range(self.n_participants_)])
         
        null =numpy.r_[self.intrasim, self.intersim]
        permuted_statistics = np.zeros(num_permutations)
        # Perform permutations
        for i in tqdm(range(num_permutations)):
            numpy.random.shuffle(null)
            permuted1 = null[:self.intrasim.size]
            permuted2 = null[self.intrasim.size:]
            permuted_statistics = permuted_statistics.at[i].set(test_statistic(permuted1, permuted2))
        observed_statistic = test_statistic(self.intrasim, self.intersim)
        # Compute p-value
        p_value = np.mean(permuted_statistics >= observed_statistic)
        return permuted_statistics, observed_statistic, p_value
    
    


