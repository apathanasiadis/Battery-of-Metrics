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
    
    def fit_occurence_ratio(self, X, y):#intersim_vs_intrasim
        '''X, y follow the standard sklearn shapes
        X: train labels
        y: test labels
        '''
        X = check_array(X)
        y = check_array(y)
        self.n_participants_ = X.shape[0]
        self.n_states = unique_labels(X.ravel()).size
        self.X_ = jax.vmap(labels2occup, in_axes=(0, None))(X, self.n_states)
        self.y_ = jax.vmap(labels2occup, in_axes=(0, None))(y, self.n_states)
        return self

    def intersim_vs_intrasim(self, num_permutations):
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
    
    def regression(self, pred_array, true_array):
        '''
        solves the Ax=b equation, where A = `X_` which can be the occurence ratio per state for instance.
        Then pred_array = `b` for which the equation is solved and can be the array of the associated ages.
        That can be solved per state and also over the linear combination of the states.
        It subsequently implements a linear regression over the `true_array`.
        returns predictors, r_values, p_values        
        '''
        p_values = np.zeros(self.n_states+1)
        r_values = np.zeros(self.n_states+1)
        predictors = np.zeros((self.n_states+1, true_array.size))
        # for each state
        for i in range(self.n_states):
            a = self.X_[:,i][:,None]
            y = self.y_[:,i][:,None] @ np.linalg.lstsq(a, pred_array, rcond=None)[0]
            assert y.size == true_array.size
            slope,_,r_value,p_value,_ = scipy.stats.linregress(x=y, y=true_array)
            p_values = p_values.at[i].set(p_value); r_values = r_values.at[i].set(r_value); 
            predictors = predictors.at[i].set(y)
        # as a linear combination of all states
        y = self.y_ @ np.linalg.lstsq(self.X_, pred_array, rcond=None)[0]
        slope,_,r_value,p_value,_ = scipy.stats.linregress(x=y, y=true_array)
        p_values = p_values.at[-1].set(p_value); r_values = r_values.at[-1].set(r_value); 
        predictors = predictors.at[-1].set(y)
        return predictors, r_values, p_values
    


