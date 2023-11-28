import numpy as np
import pandas as pd
from itertools import chain
from pyamg import smoothed_aggregation_solver
from scipy.sparse import csr_array, csr_matrix, diags, eye, lil_matrix, spmatrix, vstack, hstack, csc_matrix
from scipy.sparse.linalg import lobpcg
from sklearn.utils import check_array
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from graphai.core.common.common_utils import invert_dict


def convert_to_csr_matrix(g):
    """
    Converts a given matrix or ndarray to a CSR matrix
    :param g: Matrix or ndarray
    :return: CSR matrix
    """
    return csr_matrix(g)


def normalize_features(g):
    """
    Normalizes the rows of a matrix (according to the l2 norm)
    :param g: Dense matrix
    :return: Normalized matrix
    """
    return g / np.linalg.norm(g, ord=None, axis=1).reshape((g.shape[0], 1))


def add_eye(g):
    """
    Add the identity matrix to the given matrix
    :param g: Sparse matrix
    :return: g + I
    """
    return g + eye(g.shape[0])


def remove_eye(g):
    """
    Removes the identity matrix from the given matrix
    :param g: Sparse matrix
    :return: g - I
    """
    return g - eye(g.shape[0])


def get_matrix_power(g, p):
    """
    Computes the p-th power of a matrix
    :param g: Sparse matrix
    :param p: Power
    :return: g**p
    """
    if p == 1:
        return g
    result_left = get_matrix_power(g, p // 2)
    result_right = get_matrix_power(g, p // 2 + p % 2)
    print(p)
    return result_left.dot(result_right)


def binarize_matrix(g):
    """
    Binarizes a matrix by setting all non-zero values to 1
    :param g: Sparse matrix
    :return: Binarized matrix
    """
    g = csr_matrix(g)
    g[g >= 1] = 1
    return g


def threshold_matrix(g, thresh=3):
    """
    Thresholds a sparse matrix (everything above the threshold is kept, everything below is set to 0)
    :param g: Sparse matrix
    :param thresh: Threshold
    :return: Thresholded matrix
    """
    data = g.data.copy()
    rows, cols = g.nonzero()
    print(len(data), len(rows), len(cols))
    data[data < thresh] = 0
    rows = [rows[i] for i in range(len(rows)) if data[i] > 0]
    cols = [cols[i] for i in range(len(cols)) if data[i] > 0]
    data = [data[i] for i in range(len(data)) if data[i] > 0]
    print(len(data), len(rows), len(cols))
    return csr_matrix((data, (rows, cols)), shape=g.shape)


def binarize_matrix_by_threshold(g, thresh=3):
    """
    Binarizes the given sparse matrix using a threshold (everything above is 1 and everything below is 0)
    :param g: Sparse matrix
    :param thresh: The threshold
    :return: Binarized matrix
    """
    g = threshold_matrix(g, thresh)
    return binarize_matrix(g), g


def get_laplacian(graph, normed=False):
    """
    Computes the laplacian (normalized or unnormalized) of a graph using its adjacency matrix.
    :param graph: Adjacency matrix
    :param normed: Whether to compute the normalized laplacian
    :return: The laplacian as a LIL sparse matrix
    """
    if not isinstance(graph, csr_array):
        graph = convert_to_csr_matrix(graph)
    W = graph.copy()
    W.setdiag(0, k=0)
    d = np.array(np.sum(W, axis=0)).flatten()
    if normed:
        d = np.sqrt(d) + 1e-10
        D = diags(1/d, offsets=0)
        I = eye(W.get_shape()[0])
        L = I - D.dot(W).dot(D)
    else:
        D = diags(d, offsets=0)
        L = D - W
    return lil_matrix(L)


def sum_laplacians(laplacians):
    """
    Aggregates laplacians using a simple sum
    :param laplacians: List of laplacians
    :return: Aggregated laplacian
    """
    result = laplacians[0]
    for embedding in laplacians:
        result = result + embedding
    return result


def arithmetic_mean_laplacians(laplacians):
    """
    Aggregates laplacians using the arithmetic mean
    :param laplacians: List of laplacians
    :return: Aggregated laplacian
    """
    return sum_laplacians(laplacians) / len(laplacians)


def combine_laplacians(laplacians, mean=True):
    """
    Combines a list of matrix laplacians using the requested method
    :param laplacians: Lis of laplacians
    :param mean: whether to compute the arithmetic mean or just the sum
    :return: Aggregated laplacian
    """
    if mean:
        return arithmetic_mean_laplacians(laplacians)
    else:
        return sum_laplacians(laplacians)


def spec_embed_on_laplacian(laplacian, n_clusters, seed=420):
    """
    Computes the spectral embedding of a given laplacian matrix
    :param laplacian: The laplacian (as a sparse matrix)
    :param n_clusters: The number of components to compute
    :param seed: Random seed for the optimizer
    :return: The spectral embedding of the laplacian matrix
    """
    # This function uses code from sklearn.manifold.spectral_embedding
    diag_shift = 1e-5 * eye(laplacian.shape[0])
    laplacian += diag_shift
    ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse="csr"))
    laplacian -= diag_shift
    random_state = np.random.RandomState(seed)
    M = ml.aspreconditioner()
    # Create initial approximation X to eigenvectors
    X = random_state.standard_normal(size=(laplacian.shape[0], n_clusters))
    X = X.astype(laplacian.dtype)
    _, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.0e-5, largest=False, maxiter=30, verbosityLevel=1)
    return diffusion_map[:,:n_clusters]


def combine_and_embed_laplacian(cleanedup_main_graphs, combination_method, n_dims):
    """
    Computes and combines graph Laplacians and calculates their spectral embedding
    Arguments:
        :param cleanedup_main_graphs: List of graphs to be used
        :param combination_method: Method for combining the Laplacians, "armean" by default
        :param n_dims: Number of dimensions for the spectral embedding
    Returns:
        The combined Laplacian matrix and the spectral embedding
    """
    laplacians = [get_laplacian(cleanedup_main_graph, normed=True) for cleanedup_main_graph in cleanedup_main_graphs]
    print('Laplacians computed')
    combined_laplacian = combine_laplacians(laplacians, mean=True)
    print('Laplacians combined')
    print('Computing spectral embedding...')
    embedding = spec_embed_on_laplacian(combined_laplacian, n_dims)
    return combined_laplacian, embedding


def perform_PCA(data, n_components, random_state=420, center_and_scale=True):
    """
    Performs PCA on the data
    :param data: The original data
    :param n_components: Number of components
    :param random_state: Random state
    :return: The data after dimensionality reduction using PCA
    """
    if center_and_scale:
        data = data - np.mean(data, axis=0)
        data = data / np.std(data, axis=0)
    model = PCA(n_components=n_components, random_state=random_state, svd_solver='full')
    return model.fit_transform(data)


def precompute_clustering_metric(data, affinity, normalize_vectors, random_state):
    # Setting random seed
    np.random.seed(random_state)
    if normalize_vectors:
        data = normalize_features(data)
    if affinity == 'cosine':
        precomputed_affinity = cosine_distances(data)
        # We need to break ties in order to eliminate all non-determinism, so we add a tiny random value to
        # every cosine distance
        precomputed_affinity += 1e-10 * np.random.rand(precomputed_affinity.shape[0], precomputed_affinity.shape[1])
    else:
        precomputed_affinity = euclidean_distances(data)
    return precomputed_affinity


def perform_constrained_agglomerative(data, n_clusters, normalize_vectors=False, random_state=420,
                                      affinity='cosine', linkage='average', full_compute=False):
    """
    Performs agglomerative clustering on the data with must-link and cannot-link constraints
    :param data: The data (ndarray)
    :param n_clusters: Number of clusters
    :param ml: List of must-link constraints
    :param cl: List of cannot-link constraints
    :param normalize_vectors: Whether to normalize each data point (using l2 norm)
    :param random_state: The random state
    :param return_model: Whether to return the clustering model as well as the labels
    :param affinity: How the distance between data points is computed
    :param linkage: Linkage of the clusters
    :param full_compute: Whether to compute the full tree
    :return: Cluster labels, optionally clustering model
    """
    assert n_clusters is not None

    precomputed_affinity = precompute_clustering_metric(data, affinity, normalize_vectors, random_state)
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                            linkage=linkage, compute_full_tree=full_compute,
                                            compute_distances=False)
    predictions = cluster_model.fit_predict(precomputed_affinity)
    return predictions, cluster_model


def variance_ratio_eval(data, labels):
    """
    Computes the variance ratio of clusters
    :param data: The data
    :param labels: Cluster labels
    :return: The variance ratio score (also known as the Calinski-Harabasz score)
    """
    return calinski_harabasz_score(data, labels)


def davies_bouldin_eval(data, labels):
    """
    Computes the Davies-Bouldin score
    :param data: The data
    :param labels: Cluster labels
    :return: The score
    """
    return davies_bouldin_score(data, labels)


def cluster_using_embedding(embedding, n_clusters, params=None):
    """
    Computes one level of clustering using the provided embedding and constraints
    Arguments:
        :param embedding: Embedding of concepts in low-dimensional space
        :param params: Parameters for the clustering, e.g. # of clusters, # of PCA dimensions, etc.
    Returns:
        Clustering labels
    """

    sub_embedding = perform_PCA(embedding, params['PCA'], center_and_scale=params['scaled_pca'])
    print('PCA done')

    cluster_labels, cluster_model = perform_constrained_agglomerative(sub_embedding,
                                                                      n_clusters=n_clusters,
                                                                      normalize_vectors=params['normalize'],
                                                                      affinity=params['affinity'],
                                                                      linkage=params['linkage'],
                                                                      full_compute=False)
    print('Model trained')
    print('******************')
    # Unsupervised evaluation scores are sensitive to extreme results (e.g. everything being in one cluster),
    # so errors need to be caught here.
    try:
        print('Unsupervised evaluation scores')
        db_score = davies_bouldin_eval(sub_embedding, cluster_labels)
        print('Davies-Bouldin score (the lower the better): %f' % db_score)
        vr_score = variance_ratio_eval(sub_embedding, cluster_labels)
        print('Variance ratio (the higher the better): %f' % vr_score)
    except:
        print('Cannot compute unsupervised measures. '
              'It is likely that everything has been grouped into one single cluster.')
    return cluster_labels


def group_clustered(data, labels, mode='mean', rows_and_cols=False, precomputed_map=None):
    """
    Groups the data points together based on provided clustering labels such that each cluster becomes one
    single data point.
    :param data: The data (ndarray or sparse matrix)
    :param labels: The cluster labels
    :param mode: What to aggregate with (mean/median)
    :param rows_and_cols: Whether to perform the aggregation on both the rows and the columns of the data
    :param precomputed_map: If provided, this will be used as the cluster to concept map. If not, the mapping
                            will be computed.
    :return: The transformed data and the cluster to concept map
    """
    list_of_rows = list()
    if precomputed_map is not None:
        label_map = precomputed_map
        unique_labels = sorted(list(label_map.keys()))
    else:
        label_map = dict()
        unique_labels = sorted(list(set(labels.tolist())))
    for current_label in unique_labels:
        current_indices = np.argwhere(labels==current_label).flatten().tolist()
        if mode == 'mean':
            list_of_rows.append(data[current_indices, :].sum(axis=0) / len(current_indices))
        else:
            list_of_rows.append(np.median(data[current_indices, :], axis=0, keepdims=True))
        if precomputed_map is None:
            label_map[current_label] = current_indices
    if isinstance(data, spmatrix):
        result = vstack([csr_matrix(x) for x in list_of_rows])
    else:
        result = np.vstack(list_of_rows)
    if rows_and_cols:
        list_of_cols = list()
        for current_label in unique_labels:
            current_indices = label_map[current_label]

            if mode == 'mean':
                list_of_cols.append(result[:, current_indices].sum(axis=1) / len(current_indices))
            else:
                list_of_cols.append(np.median(result[:, current_indices], axis=1, keepdims=True))
        if isinstance(data, spmatrix):
            result = hstack([csc_matrix(x) for x in list_of_cols])
        else:
            result = np.hstack(list_of_cols)
    return result, label_map


def reassign_outliers(labels, embeddings, min_n=3):
    """
    Reassigns outlier clusters to non-outlier clusters.
    Arguments:
        :param labels: Labels of each concept
        :param embeddings: The concept embedding vectors
        :param min_n: Minimum size for a cluster to not be considered an outlier
    Returns:
        New labels after reassignment
    """
    df = pd.DataFrame({'label': labels, 'page_index': list(range(len(labels)))})
    print(df)
    print('Computing sets of outlying and non-outlying concepts and clusters')
    outlying_cluster_set = df.groupby('label').count().reset_index()
    outlying_cluster_set = set(outlying_cluster_set.loc[
                               outlying_cluster_set['page_index'] < min_n]['label'].values.tolist())
    if len(outlying_cluster_set) == 0:
        return labels
    non_outlying_cluster_set = set(labels).difference(outlying_cluster_set)
    print(outlying_cluster_set)
    outlying_concept_set = set(
        df.loc[df['label'].apply(lambda x: x in outlying_cluster_set)]['page_index'].values.tolist()
    )

    similarity_map = embeddings.dot(embeddings.transpose())

    print('Computing outlier and non-outlier embeddings')
    similarity_map, cluster_map = group_clustered(
        similarity_map, np.array(labels), mode='median', rows_and_cols=True
    )

    outlying_map = {x: cluster_map[x] for x in outlying_cluster_set}
    non_outlying_map = {x: cluster_map[x] for x in non_outlying_cluster_set}
    outlying_concept_to_cluster_map = chain.from_iterable([[(y,x) for y in outlying_map[x]] for x in outlying_map])
    outlying_concept_to_cluster_map = {x[0]:x[1] for x in outlying_concept_to_cluster_map}
    outlying_cluster_labels = sorted(list(outlying_map.keys()))
    outlying_cluster_labels_inverse = invert_dict(dict(enumerate(outlying_cluster_labels)))
    non_outlying_cluster_labels = sorted(list(non_outlying_map.keys()))
    non_outlying_cluster_labels_dict = dict(enumerate(non_outlying_cluster_labels))
    non_outlying_cluster_labels_inverse = invert_dict(non_outlying_cluster_labels_dict)

    similarity_map = similarity_map[outlying_cluster_labels, :][:, non_outlying_cluster_labels]
    if isinstance(similarity_map, np.ndarray):
        closest_cluster_to_outlying_cluster = np.argmax(similarity_map, axis=1).flatten()
    else:
        closest_cluster_to_outlying_cluster = np.array(similarity_map.argmax(axis=1)).flatten()
    closest_cluster_to_outlying_cluster = [int(x) for x in closest_cluster_to_outlying_cluster]
    # If the concept is part of the outlying concepts, we do the following:
    # 1. Get the label of the corresponding cluster.
    # 2. Get the index of that cluster for the aggregated embedding
    # 3. Get the index of the closest non-outlying cluster
    # 4. Get the label of the latter cluster
    print('Computing new labels')
    new_labels = [labels[i] if i not in outlying_concept_set
                  else non_outlying_cluster_labels_dict[
                            closest_cluster_to_outlying_cluster[
                                outlying_cluster_labels_inverse[
                                    outlying_concept_to_cluster_map[i]]]]
                  for i in range(len(labels))]
    new_labels = np.array([non_outlying_cluster_labels_inverse[x] for x in new_labels])

    return new_labels
