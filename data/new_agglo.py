"""
Unused except for _hc_cut.
Agglomerative Clustering, courtesy of https://stackoverflow.com/questions/26851553/sklearn-agglomerative-clustering-linkage-matrix
"""

from heapq import heapify, heappop, heappush, heappushpop
import warnings
import sys

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from sklearn.utils.validation import check_array
from sklearn.utils.sparsetools import connected_components
from sklearn.cluster import _hierarchical
from sklearn.cluster.hierarchical import ward_tree
from sklearn.cluster._feature_agglomeration import AgglomerationTransform
from sklearn.utils.fast_dict import IntFloatDict


def _fix_connectivity(X, connectivity, n_components=None,
                      affinity="euclidean"):
    """
    Fixes the connectivity matrix
        - copies it
        - makes it symmetric
        - converts it to LIL if necessary
        - completes it if necessary
    """
    n_samples = X.shape[0]
    if (connectivity.shape[0] != n_samples or
        connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s '
                         'when X is %s' % (connectivity.shape, X.shape))

    # Make the connectivity matrix symmetric:
    connectivity = connectivity + connectivity.T

    # Convert connectivity matrix to LIL
    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_components, labels = connected_components(connectivity)

    if n_components > 1:
        warnings.warn("the number of connected components of the "
                      "connectivity matrix is %d > 1. Completing it to avoid "
                      "stopping the tree early." % n_components,
                      stacklevel=2)
        # XXX: Can we do without completing the matrix?
        for i in xrange(n_components):
            idx_i = np.where(labels == i)[0]
            Xi = X[idx_i]
            for j in xrange(i):
                idx_j = np.where(labels == j)[0]
                Xj = X[idx_j]
                D = pairwise_distances(Xi, Xj, metric=affinity)
                ii, jj = np.where(D == np.min(D))
                ii = ii[0]
                jj = jj[0]
                connectivity[idx_i[ii], idx_j[jj]] = True
                connectivity[idx_j[jj], idx_i[ii]] = True

    return connectivity, n_components

# average and complete linkage
def linkage_tree(X, connectivity=None, n_components=None,
                 n_clusters=None, linkage='complete', affinity="euclidean",
                 return_distance=False):
    """Linkage agglomerative clustering based on a Feature matrix.
    The inertia matrix uses a Heapq-based representation.
    This is the structured version, that takes into account some topological
    structure between samples.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        feature matrix representing n_samples samples to be clustered
    connectivity : sparse matrix (optional).
        connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.
    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.
        NOTE: This parameter is now directly determined directly
        from the connectivity matrix and will be removed in 0.18
    n_clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.
    linkage : {"average", "complete"}, optional, default: "complete"
        Which linkage critera to use. The linkage criterion determines which
        distance to use between sets of observation.
            - average uses the average of the distances of each observation of
              the two sets
            - complete or maximum linkage uses the maximum distances between
              all observations of the two sets.
    affinity : string or callable, optional, default: "euclidean".
        which metric to use. Can be "euclidean", "manhattan", or any
        distance know to paired distance (see metric.pairwise)
    return_distance : bool, default False
        whether or not to return the distances between the clusters.
    Returns
    -------
    children : 2D array, shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`
    n_components : int
        The number of connected components in the graph.
    n_leaves : int
        The number of leaves in the tree.
    parents : 1D array, shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.
    distances : ndarray, shape (n_nodes-1,)
        Returned when return_distance is set to True.
        distances[i] refers to the distance between children[i][0] and
        children[i][1] when they are merged.
    See also
    --------
    ward_tree : hierarchical clustering with ward linkage
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    linkage_choices = {'complete': _hierarchical.max_merge,
                       'average': _hierarchical.average_merge,
                      }
    try:
        join_func = linkage_choices[linkage]
    except KeyError:
        raise ValueError(
            'Unknown linkage option, linkage should be one '
            'of %s, but %s was given' % (linkage_choices.keys(), linkage))

    if connectivity is None:
        from scipy.cluster import hierarchy  # imports PIL

        if n_clusters is not None:
            warnings.warn('Partial build of the tree is implemented '
                          'only for structured clustering (i.e. with '
                          'explicit connectivity). The algorithm '
                          'will build the full tree and only '
                          'retain the lower branches required '
                          'for the specified number of clusters',
                          stacklevel=2)

        if affinity == 'precomputed':
            # for the linkage function of hierarchy to work on precomputed
            # data, provide as first argument an ndarray of the shape returned
            # by pdist: it is a flat array containing the upper triangular of
            # the distance matrix.
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        elif affinity == 'l2':
            # Translate to something understood by scipy
            affinity = 'euclidean'
        elif affinity in ('l1', 'manhattan'):
            affinity = 'cityblock'
        elif callable(affinity):
            X = affinity(X)
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        out = hierarchy.linkage(X, method=linkage, metric=affinity)
        children_ = out[:, :2].astype(np.int)

        if return_distance:
            distances = out[:, 2]
            return children_, 1, n_samples, None, distances
        return children_, 1, n_samples, None

    if n_components is not None:
        warnings.warn(
            "n_components is now directly calculated from the connectivity "
            "matrix and will be removed in 0.18",
            DeprecationWarning)
    connectivity, n_components = _fix_connectivity(X, connectivity)

    connectivity = connectivity.tocoo()
    # Put the diagonal to zero
    diag_mask = (connectivity.row != connectivity.col)
    connectivity.row = connectivity.row[diag_mask]
    connectivity.col = connectivity.col[diag_mask]
    connectivity.data = connectivity.data[diag_mask]
    del diag_mask

    if affinity == 'precomputed':
        distances = X[connectivity.row, connectivity.col]
    else:
        # FIXME We compute all the distances, while we could have only computed
        # the "interesting" distances
        distances = paired_distances(X[connectivity.row],
                                     X[connectivity.col],
                                     metric=affinity)
    connectivity.data = distances

    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters

    if return_distance:
        distances = np.empty(n_nodes - n_samples)
    # create inertia heap and connection matrix
    A = np.empty(n_nodes, dtype=object)
    inertia = list()

    # LIL seems to the best format to access the rows quickly,
    # without the numpy overhead of slicing CSR indices and data.
    connectivity = connectivity.tolil()
    # We are storing the graph in a list of IntFloatDict
    for ind, (data, row) in enumerate(zip(connectivity.data,
                                          connectivity.rows)):
        A[ind] = IntFloatDict(np.asarray(row, dtype=np.intp),
                              np.asarray(data, dtype=np.float64))
        # We keep only the upper triangular for the heap
        # Generator expressions are faster than arrays on the following
        inertia.extend(_hierarchical.WeightedEdge(d, ind, r)
                       for r, d in zip(row, data) if r < ind)
    del connectivity

    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=np.intp)
    children = []

    # recursive merge loop
    for k in xrange(n_samples, n_nodes):
        # identify the merge
        while True:
            edge = heappop(inertia)
            if used_node[edge.a] and used_node[edge.b]:
                break
        i = edge.a
        j = edge.b

        if return_distance:
            # store distances
            distances[k - n_samples] = edge.weight

        parent[i] = parent[j] = k
        children.append((i, j))
        # Keep track of the number of elements per cluster
        n_i = used_node[i]
        n_j = used_node[j]
        used_node[k] = n_i + n_j
        used_node[i] = used_node[j] = False

        # update the structure matrix A and the inertia matrix
        # a clever 'min', or 'max' operation between A[i] and A[j]
        coord_col = join_func(A[i], A[j], used_node, n_i, n_j)
        for l, d in coord_col:
            A[l].append(k, d)
            # Here we use the information from coord_col (containing the
            # distances) to update the heap
            heappush(inertia, _hierarchical.WeightedEdge(d, k, l))
        A[k] = coord_col
        # Clear A[i] and A[j] to save memory
        A[i] = A[j] = 0

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples

    # # return numpy array for efficient caching
    children = np.array(children)[:, ::-1]

    if return_distance:
        return children, n_components, n_leaves, parent, distances
    return children, n_components, n_leaves, parent

# Matching names to tree-building strategies
def _complete_linkage(*args, **kwargs):
    kwargs['linkage'] = 'complete'
    return linkage_tree(*args, **kwargs)

def _average_linkage(*args, **kwargs):
    kwargs['linkage'] = 'average'
    return linkage_tree(*args, **kwargs)

_TREE_BUILDERS = dict(
    ward=ward_tree,
    complete=_complete_linkage,
    average=_average_linkage,
    )

def _hc_cut(n_clusters, children, n_leaves):
    """Function cutting the ward tree for a given number of clusters.
    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.
    children : list of pairs. Length of n_nodes
        The children of each non-leaf node. Values less than `n_samples` refer
        to leaves of the tree. A greater value `i` indicates a node with
        children `children[i - n_samples]`.
    n_leaves : int
        Number of leaves of the tree.
    Returns
    -------
    labels : array [n_samples]
        cluster labels for each point
    """
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree
    nodes = [-(max(children[-1]) + 1)]
    for i in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label

class NewAgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    Agglomerative Clustering
    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.
    affinity : string or callable, default: "euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or 'precomputed'.
        If linkage is "ward", only "euclidean" is accepted.
    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.
        NOTE: This parameter is now directly determined from the connectivity
        matrix and will be removed in 0.18
    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree.
    linkage : {"ward", "complete", "average"}, optional, default: "ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument ``axis=1``, and reduce it to an array of size [M].
    Attributes
    ----------
    labels_ : array [n_samples]
        cluster labels for each point
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_components_ : int
        The estimated number of connected components in the graph.
    children_ : array-like, shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`
    """

    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, n_components=None,
                 compute_full_tree='auto', linkage='ward',
                 pooling_func=np.mean):
        self.n_clusters = n_clusters
        self.memory = memory
        self.n_components = n_components
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.pooling_func = pooling_func

    def fit(self, X, y=None):
        """Fit the hierarchical clustering on the data
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.
        Returns
        -------
        self
        """
        X = check_array(X)[0]
        memory = self.memory
        if isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError("%s was provided as affinity. Ward can only "
                             "work with euclidean distances." %
                             (self.affinity, ))

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type %s."
                             "Valid options are %s" % (self.linkage,
                                                       _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(
                connectivity, accept_sparse=['csr', 'coo', 'lil'])

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            # Early stopping is likely to give a speed up only for
            # a large number of clusters. The actual threshold
            # implemented here is heuristic
            compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}
        kwargs['return_distance'] = True
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity
        self.children_, self.n_components_, self.n_leaves_, parents, \
            self.distance = memory.cache(tree_builder)(X, connectivity,
                                       n_components=self.n_components,
                                       n_clusters=n_clusters,
                                       **kwargs)
        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters, self.children_,
                                   self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reasign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self
