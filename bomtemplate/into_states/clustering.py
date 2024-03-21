import jax
import jax.numpy as np
from jax._src.numpy.util import check_arraylike
import numpy
import scipy.sparse.csgraph as spgraph
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
### for consensus algorithm
# from netneurotools import cluster
# from joblib import Parallel, delayed
# jax.config.update('jax_platform_name', 'cpu')
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'

def get_similarity(centroids):
    '''centroids.shape=(n_clusters, n_nodes)'''
    centroids /= np.linalg.norm(centroids, axis=1)[:,None]
    similarity = np.abs(centroids @ centroids.T)
    return similarity

def get_laplacian_eigenvs(v1s, n_init_clusters):
    '''
    Inputs
    ---------------------------
    v1s: array
        v1s.shape = (n_participants, n_samples, n_nodes)
        or (n_concat_samples, n_nodes)
    n_init_clusters: int

    Outputs:
    ----------------------------
    (init_centroids, init_labels): tuple
        contains the initial minibatchkmeans results over the v1s
    (k_evals, k_evecs): tuple
        contains the first k eigenvalues and eigenvectors of the graph laplacian.
        the `i` eigenvalue corresponds to the k_evecs[:,i] eigenvector,
        following the scheme of `np.linalg.eigh`.
    '''
    assert v1s.ndim == 2
    km = MiniBatchKMeans(n_clusters=n_init_clusters, n_init='auto')#, random_state=rs)
    # print(v1s.shape)
    km.fit(v1s)
    init_centroids = km.cluster_centers_
    init_labels = km.labels_
    similarity = get_similarity(init_centroids)
    graph = spgraph.csgraph_from_dense(similarity)
    L = spgraph.laplacian(graph, normed=True, copy=True)
    k_evals, k_evecs = np.linalg.eigh(L.todense())
    return km, (init_centroids, init_labels), (k_evals, k_evecs)


class LeidaCenSC(BaseEstimator):

    def __init__(self, demo_param='demo_param', n_states=None, n_init_clusters=100):
        self.demo_param = demo_param
        self.n_states = n_states
        self.n_init_clusters = n_init_clusters
    
    def fit(self, X, y=None):
        if X.ndim == 3:
            n_participants, n_samples, self.n_nodes = X.shape
            X = X.reshape(-1, self.n_nodes)
        elif X.ndim == 2:
            n_concat_samples, self.n_nodes = X.shape
        else:
            raise ValueError('worng `X` (namely `v1s`) input dimenions')
        assert X.ndim == 2
        # Return the transformer
        self.km, init_km, k_evs = get_laplacian_eigenvs(X, n_init_clusters=self.n_init_clusters) # random initializations matter within minibatchkmeans for getting the centorids
        self.init_centroids, self.init_labels = init_km
        self.k_evals, k_evecs = k_evs
        return self, k_evecs

    def transform(self, k_evecs):
        '''feed the first k eigenvectors from `fit` output (k_evecs) to get the final labels'''
        check_is_fitted(self, 'km')
        n_states_calculated = np.argsort(np.diff(np.diff(self.k_evals[1:])))[:4] + 1 + 1 # the best two candidates
        if self.n_states == None or self.n_states == 'best':
            self.n_states = n_states_calculated[0]
        elif self.n_states == 'max_n_states':
            self.n_states = n_states_calculated.max()
        self.n_states = int(self.n_states)
        se_vectors = k_evecs[:,1:self.n_states+1].copy()               # ignore 0th vector for the spectral embedding
        se_vectors /= np.linalg.norm(se_vectors, axis=-1)[:,None]  # get the spectral embedding vectors
        # and apply kmeans over the spectral embedding
        km_se = KMeans(n_clusters=self.n_states, n_init='auto').fit(se_vectors)        
        se_labels = km_se.labels_.astype('i')
        se_centroids = km_se.cluster_centers_
        # this clusters the centroids into n_states
        self.states_centroids_dict = {value: numpy.where(se_labels == value)[0] for value in numpy.unique(se_labels)}
        # subsequently, this assigns the init_labels to the clustered centroids
        states_trainsamples_dict = {v: np.nonzero(np.isin(self.init_labels, k))[0] for v,k in self.states_centroids_dict.items()}
        # get train_labels from the states_samples_dict
        self.labels_ = numpy.empty_like(self.init_labels)
        for v,k in states_trainsamples_dict.items(): 
            self.labels_[k] = v
        return self.labels_

    def predict(self, X):
        check_is_fitted(self, 'labels_')
        if X.ndim == 3:
            n_participants, n_samples, n_nodes_ = X.shape
            X = X.reshape(-1, n_nodes_)
        elif X.ndim == 2:
            n_concat_samples, n_nodes_ = X.shape
        else:
            raise ValueError('worng `X` (namely `v1s`) input dimenions')
        assert n_nodes_ == self.n_nodes
        test_labels = self.km.predict(X)
        states_testsamples_dict = {v: np.nonzero(np.isin(test_labels, k))[0] for v,k in self.states_centroids_dict.items()}
        test_labels_ = numpy.empty_like(test_labels)
        for v,k in states_testsamples_dict.items(): 
            test_labels_[k] = v
        return test_labels_
    
    

class Cluster(ClassifierMixin, TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, clusterclass):
        '''fit using a class instance from sklearn.cluster'''
        X = check_array(X, accept_sparse=True)
        self.estimator = clusterclass.fit(X)
        return self
    
    def transform(self, X):
        '''
        embed the data to some manifold
        returns the embedded data
        '''
        # Check is fit had been called
        check_is_fitted(self, 'estimator')
        # Input validation
        X = check_array(X, accept_sparse=True)
        try:
            X_ = self.estimator.transform(X)
        except AttributeError:
            X_ = self.estimator.fit_transform(X)
        return X_

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'estimator')
        self.labels_ = self.estimator.predict(X)
        return self.labels_
    
    def evaluation(self, X, sample_size=None):
        '''get the average silhouette score using `metrics.silhouette_score`'''
        check_is_fitted(self, 'estimator')
        silh_sc = metrics.silhouette_score(
            X,
            self.labels_,
            metric="euclidean",
            sample_size=sample_size
            )
        return silh_sc