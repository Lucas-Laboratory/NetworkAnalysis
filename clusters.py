import numpy as np

from itertools import permutations

import markov_clustering as mc

import funcs


def markov_clusters(adj_mat, inflation):
    '''
    Compute clusters on an adjacency matrix using the Markov Clustering algorithm

    Calls markov_clustering.run_mcl to compute clusters. Then returns cluster 
    membership for each node in adj_mat

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    inflation (float) : inflation parameter passed to run_mcl

    Returns
    -------
    clusters (list) : List of numbers in order of adj_mat entries, indicating
                      cluster memberships for each node
    '''
    result = mc.run_mcl(adj_mat, inflation=inflation)
    clusters = mc.get_clusters(result)
    return clusters

def delta_matrix(matrix, clusters):
    """
    Compute delta matrix where delta[i,j]=1 if i and j belong
    to same cluster and i!=j
    
    Parameters
    ----------
    matrix (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    clusters (list) : The clusters returned by markov_clusters

    Returns
    -------
    delta (numpy.array) : delta matrix
    """
    delta = np.zeros(matrix.shape)

    for i in clusters :
        for j in permutations(i, 2):
            delta[j] = 1

    return delta

def modularity(matrix, clusters):
    """
    Compute Q value for modularity of cluster assignments within a matrix

    Used for assessing optimal clustering using 'test_modularity()'
    
    Parameters
    ----------
    matrix (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    clusters (list) : The clusters returned by markov_clusters

    Returns
    -------
    Q (numpy.array) : modularity metric
    """     
    m = matrix.sum()

    matrix_2 = matrix

    expected = lambda i,j: (matrix_2[i,:].sum()*matrix[:,j].sum() )

    delta   = delta_matrix(matrix, clusters)
    indices = np.array(delta.nonzero())
    
    Q = sum( matrix[i, j] - expected(i, j)/m for i, j in indices.T )/m
    
    return Q

def test_modularity(matrix, 
                    inflations=[i / 10 for i in range(11, 46)],
                    expansions=[2]):
    '''
    Tests clustering quality for Markov clustering with multiple
    inflation values

    Calls markov_clusters() to compute clusters, for each inflation value
    in inflations. Then returns cluster list of Q values for all inflations

    Parameters
    ----------
    matrix (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    inflations (list(float)) : inflation parameter passed to run_mcl

    Returns
    -------
    Qs (list) : List of modularity metric values for each inflation value
    '''
    Qs = np.empty((len(inflations), len(expansions)))
    for i, inflation in enumerate(inflations):
        for j, expansion in enumerate(expansions):
            result = mc.run_mcl(matrix, inflation=inflation, expansion=expansion)
            clusters = mc.get_clusters(result)
            Q = modularity(matrix=result, clusters=clusters)
            # print("inflation:", inflation, "expansion:", expansion, "modularity:", Q)
            Qs[i, j] = Q

    return Qs

def get_cluster_arr(cluster_list_of_tuples):
    '''
    Convert cluster info from a list of tuples, as provided by the 
    markov_clustering.run_mcl().get_clusters() function, into a
    numpy array the length of the list of Regions (data.index), 
    where each entry is cluster index

    Parameters
    ----------
    cluster_list_of_tuples (List(tuple)) : List of (region_number, cluster_ind)

    Returns
    -------
    clusters_arr (numpy.array) : Array of cluster assignments, sorted by Region index
    '''
    clusters_arr = np.empty((np.sum([len(cluster) for cluster in cluster_list_of_tuples])))
    # Sort cluster_list_of_tuples by sizes of tuples 
    # to set cluster number based on size of cluster
    sorted_cluster_list_of_tuples = sorted(cluster_list_of_tuples, key=len, reverse=True)
    for c, ns in enumerate(sorted_cluster_list_of_tuples):
        for n in ns:
            clusters_arr[n] = c
    return clusters_arr

def rearrange_corr_df_by_clusters(corr_df, clusters_arr):
    '''
    Rearrange rows and columns of a correlation DataFrame by cluster order

    Parameters
    ----------
    corr_df (pandas.DataFrame) : DataFrame of correlation coefficients, can be 
                                 found with funcs.get_cross_correlation_matrix
    clusters_arr (numpy.array) : Array of cluster assignments, sorted by Region index
                                 found with get_cluster_arr

    Returns
    -------
    corr_df_soted (pandas.DataFrame) : DataFrame of correlation coefficients, 
                                       sorted by cluster assignments
    '''
    indices = np.argsort(clusters_arr)
    corr_df_sorted = corr_df[corr_df.index[indices]].iloc[indices]
    return corr_df_sorted

def rearrange_adj_mat_by_clusters(adj_mat, clusters_arr):
    '''
    Rearrange rows and columns of an adjacency matrix numpy.array by cluster order

    Parameters
    ----------
    adj_mat (numpy.array) : Adjacency matrix found with funcs.get_adjacency_matrix
    clusters_arr (numpy.array) : Array of cluster assignments, sorted by Region index
                                 found with get_cluster_arr

    Returns
    -------
    adj_mat_soted (numpy.array) : Adjacency matrix, sorted by cluster assignments
    '''
    indices = np.argsort(clusters_arr)
    adj_mat_sorted = adj_mat[indices][:,indices]
    return adj_mat_sorted

def get_cluster_ids_for_groups(data, groups, threshold_type='p', threshold=0.05, method='pearson'):
    '''
    Compute cluster IDs on a dataset for all groups

    Wrapper to start from c-fos cell count data, compute adjacency matrix for 
    each group, then run markov clustering for each group with multiple
    inflation values, and return cluster assignments

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold value to compute adj_mats for
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_adjacency_matrix

    Returns
    -------
    cluster_ids_dict (dict{string: numpy.array}) : Dictionary of cluster assignments
                                                   for each group:
                                                    {'group_name': clusters_arr}
    '''
    adj_mat_dict = funcs.get_adj_mat_dict_for_groups(data, 
                                                     groups, 
                                                     threshold_type=threshold_type, 
                                                     threshold=threshold,
                                                     method=method
                                                    )
    cluster_ids_dict = {}
    for group in groups:
        cluster_ids_dict[group] = get_cluster_ids(adj_mat_dict[group])
    return cluster_ids_dict

def get_cluster_ids(adj_mat):
    '''
    Compute cluster assignments for an adjacency matrix

    Uses inflation values from 1.1 to 30.6 and finds 'optimal' inflation value
    for Markov Clustering by finding the higher modularity metric

    Parameters
    ----------
    adj_mat (numpy.array) : Adjacency matrix found with funcs.get_adjacency_matrix

    Returns
    -------
    clusters_arr (numpy.array) : Array of cluster assignments in order of 
                                 rows of adj_mat
    '''
    inflations = [i / 10 for i in range(11, 306)]
    expansions = [2]
    Qs = test_modularity(adj_mat, inflations=inflations, expansions=expansions)
    clusters = markov_clusters(adj_mat, inflations[np.argmax(Qs)])
    clusters_arr = get_cluster_arr(clusters)
    return clusters_arr

def get_node_to_cluster(cluster_ids):
    '''
    Convert cluster array into a dictionary of node index: cluster pairs
    
    Parameters
    ----------
    cluster_ids (list(int) or numpy.array) : iterable of ordered cluster ids

    Returns
    -------
    node_to_cluster (dict{int: int}) : Dict of {node index: cluster id} pairs
    '''
    node_to_cluster = dict(zip(range(len(cluster_ids)), cluster_ids))
    return node_to_cluster

def get_cluster_to_nodes(cluster_ids):
    '''
    Convert cluster array into a dictionary of cluster id: node index pairs
    
    Parameters
    ----------
    cluster_ids (list(int) or numpy.array) : iterable of ordered cluster ids

    Returns
    -------
    node_to_cluster (dict{int: int}) : Dict of {cluster id: node index} pairs
    '''
    node_to_cluster = get_node_to_cluster(cluster_ids=cluster_ids)
    cluster_to_nodes = dict()
    for key, value in node_to_cluster.items():
        cluster_to_nodes.setdefault(value, set()).add(key)
    return cluster_to_nodes  

def trim_disconnected_nodes(data, adj_mat, cluster_ids, min_size=2):
    '''
    Remove nodes from data sets that are in small clusters

    Finds clustsers of size < min_size, and removes nodes in those clustsers
    from data, adj_mat and cluster_ids

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    adj_mat (numpy.array) : Adjacency matrix found with funcs.get_adjacency_matrix
    cluster_ids (list(int) or numpy.array) : iterable of ordered cluster ids

    Returns
    -------
    trimmed_data (pandas.DataFrame) : trimmed data frame of input data
    trimmed_adj_mat (numpy.array) : trimmed adjacency matrix 
    trimmed_cluster_ids (list(int) or numpy.array) : trimmed iterable of 
                                                     ordered cluster ids
    '''
    cluster_to_nodes = get_cluster_to_nodes(cluster_ids)
    nodes_to_trim = []
    for cluster_id, cluster_nodes in cluster_to_nodes.items():
        if len(cluster_nodes) < min_size:
            nodes_to_trim.extend(cluster_nodes)
    trimmed_data = data.drop(data.index[nodes_to_trim])
    trimmed_adj_mat = np.delete(np.delete(adj_mat, nodes_to_trim, axis=0), nodes_to_trim, axis=1)
    trimmed_cluster_ids = np.delete(cluster_ids, nodes_to_trim)
    return trimmed_data, trimmed_adj_mat, trimmed_cluster_ids
