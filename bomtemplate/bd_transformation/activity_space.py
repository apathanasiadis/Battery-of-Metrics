import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from bomtemplate import utility


class Thresholding(TransformerMixin, BaseEstimator):
    """
    Thresholding is performed by using the utility.relu function,
    that zeroes the activity with amplitude < the given threshold
    and keeps intact the rest of the activity. 
    Inspired by https://www.science.org/doi/pdf/10.1126/sciadv.abq8566;
    Peng et al. 2023 science advances
    """

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param
    
    def fit(self, X, y=None):
        if X.ndim == 3:
            self.n_participants, self.n_samples, self.n_features = X.shape      
        else:
            raise ValueError('ndim of input != 3.')
        return self

    def transform(self, X, threshold):
        """ 
        Parameters
        ----------
        X : 3d

        Returns
        -------
        X_transformed : 3d
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features')
        X_transformed = utility.relu(X, h=threshold)
        return X_transformed