import jax
import jax.numpy as np
import numpy
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

class Predict(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        '''X, y follow the standard sklearn shapes
        X shape (samples, features)
        X: “Coefficient” matrix; X = self.X_, that for example carries the occurence ratios for n states (n columns).
        y: Ordinate or “dependent variable” values; y = pred_array

        Return
        ------
        transformer
        '''
        # Check that X and y have correct shape
        X = check_array(X)
        self.x, self.residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return self
    
    def predict(self, test_array, true_array):
        '''perform linear regression between the '''
        check_is_fitted(self, 'x')
        predictor = test_array @ self.x
        _,_,r_value,p_value,_ = scipy.stats.linregress(x=predictor, y=true_array) # score
        return predictor, r_value, p_value
    

    # def regression(self, pred_array, true_array):
    #     '''
    #     solves the Ax=b equation, where A = `X_` which can be the occurence ratio per state for instance.
    #     Then pred_array = `b` for which the equation is solved and can be the array of the associated ages.
    #     That can be solved per state and also over the linear combination of the states.
    #     It subsequently implements a linear regression over the `true_array`.
    #     returns predictors, r_values, p_values        
    #     '''
    #     p_values = np.zeros(self.n_states+1)
    #     r_values = np.zeros(self.n_states+1)
    #     predictors = np.zeros((self.n_states+1, true_array.size))
    #     # for each state
    #     for i in range(self.n_states):
    #         a = self.X_[:,i][:,None]
    #         y = self.y_[:,i][:,None] @ np.linalg.lstsq(a, pred_array, rcond=None)[0]
    #         assert y.size == true_array.size
    #         slope,_,r_value,p_value,_ = scipy.stats.linregress(x=y, y=true_array)
    #         p_values = p_values.at[i].set(p_value); r_values = r_values.at[i].set(r_value); 
    #         predictors = predictors.at[i].set(y)
    #     # as a linear combination of all states -- into a new separate class
    #     y = self.y_ @ np.linalg.lstsq(self.X_, pred_array, rcond=None)[0] # fit_predict - keep [1] - as attributes - the residuals = sum squared error to compare the different regressors, like pca fit -> transform
    #     slope,_,r_value,p_value,_ = scipy.stats.linregress(x=y, y=true_array) # score
    #     p_values = p_values.at[-1].set(p_value); r_values = r_values.at[-1].set(r_value); 
    #     predictors = predictors.at[-1].set(y)
    #     return predictors, r_values, p_values