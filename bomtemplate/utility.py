import numpy as np
import jax.numpy as jp

# transformations
relu = lambda x,h: (x*(np.abs(x)>h)) # sort of binarization

# functional connectivity
def get_anchor_node(X):
    assert X.ndim == 3
    n_participants, n_samples, n_features = X.shape
    FCs_mean = jp.zeros((n_features, n_features))
    for i in range(n_participants):
        FCs_mean += jp.abs(jp.corrcoef(X[i,...], rowvar=False))
    FCs_mean /= n_participants
    anchor = int(jp.argmax(FCs_mean.sum(1)))
    return anchor

# statistics
def z_score(x):
    return (x-x.mean())/x.std()

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

# masks
def make_mask(n, indices):
    '''
    make a mask matrix with given indices

    Parameters
    ----------
    n : int
        size of the mask matrix
    indices : list
        indices of the mask matrix

    Returns
    -------
    mask : numpy.ndarray
        mask matrix
    '''
    # check validity of indices
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError('indices must be a list, tuple, or numpy array.')
    if not all(isinstance(i, int) for i in indices):
        raise ValueError('indices must be a list of integers.')
    if not all(i < n for i in indices):
        raise ValueError('indices must be smaller than n.')

    mask = np.zeros((n, n))
    mask[np.ix_(indices, indices)] = 1

    return mask


def get_intrah_mask(n_nodes):
    '''
    Get a mask for intrahemispheric connections.

    Inputs
    ------------
    n_nodes: int
        number of total nodes that constitute the data.

    Outputs
    ------------
    mask_intrah: 2d array
        mask for intrahemispheric connections.
    '''
    row_idx = np.arange(n_nodes)
    idx1 = np.ix_(row_idx[:n_nodes//2], row_idx[:n_nodes//2])
    idx2 = np.ix_(row_idx[n_nodes//2:], row_idx[n_nodes//2:])
    # build on a zeros mask
    mask_intrah = np.zeros((n_nodes, n_nodes))
    mask_intrah[idx1] = 1
    mask_intrah[idx2] = 1
    return mask_intrah


def get_interh_mask(n_nodes):
    '''
    Get a mask for interhemispheric connections.

    Inputs
    ------------
    n_nodes: int
        number of total nodes that constitute the data.

    Outputs
    ------------
    mask_interh: 2d array
        mask for interhemispheric connections.
    '''
    row_idx = np.arange(n_nodes//2)
    col_idx1 = np.where(np.eye(n_nodes, k=-n_nodes//2))[0]
    col_idx2 = np.where(np.eye(n_nodes, k=n_nodes//2))[0]
    idx1 = np.ix_(row_idx, col_idx1)
    idx2 = np.ix_(row_idx+n_nodes//2, col_idx2)
    # build on a zeros mask
    mask_interh = np.zeros((n_nodes, n_nodes))
    mask_interh[idx1] = 1
    mask_interh[idx2] = 1
    return mask_interh


def get_masks(n_nodes, networks):
    '''
    Get a dictionary of masks based on the requested networks.

    Parameters
    ------------
    n_nodes: int
        number of total nodes that constitute the data.
    networks: list of str
        list of networks to be included in the dictionary.
        'full': full-network connections
        'intrah': intrahemispheric connections
        'interh': interhemispheric connections
        to get a custom mask with specific indices
        refere to `hbt.utility.make_mask(n, indices)`.

    Outputs
    ------------
    masks: dict
        dictionary of masks based on the requested networks.
    '''
    masks = {}
    valid_networks = ['full', 'intrah', 'interh']

    for i, ntw in enumerate(networks):
        if ntw not in valid_networks:
            raise ValueError(
                f"Invalid network: {ntw}. Please choose from {valid_networks}.")
        if ntw == 'full':
            masks[ntw] = np.ones((n_nodes, n_nodes))
        elif ntw == 'intrah':
            masks[ntw] = get_intrah_mask(n_nodes)
        elif ntw == 'interh':
            masks[ntw] = get_interh_mask(n_nodes)

    return masks