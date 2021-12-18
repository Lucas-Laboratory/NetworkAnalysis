import scipy
import numpy as np

from sklearn.metrics import r2_score
from sklearn.neighbors import KernelDensity

import seaborn as sns
import pandas as pd
import networkx as nx

import bct

paper_folder = 'du_Plessis_et_al_2022/'

def load_data(
        filename=paper_folder+'Fall 2021 Compiled Data 10_9_21.xlsx',
        labels_to_drop=['Unnamed: 0']):
    '''
    Loads xlsx spreadsheet of c-fos positive cell counts

    Notes
    -----
    Currently has n per group and group names hardcoded

    Parameters
    ----------
    filename (string) : Full path to xlsx file
    labels_to_drop (List(string))) : Columns to drop.
                                     Defaults sets first column as 'Region' label, 
                                     then drops first column from main data

    Returns
    -------
    data (pandas.DataFrame) : Pandas Dataframe of c-fos count data
                              Columns are MultiIndex with 2 levels:
                              'group', and mouse name
                              Rows are labelled with 'Region'
                              Access all data for a group with e.g. data['naive_male']
    '''
    # labels_to_drop for original dataset: 
    # ['Unnamed: 0',  
    #                   'Unnamed: 15',
    #                   'Unnamed: 17',
    #                   'Unnamed: 18',
    #                   'interaction in two-way ANOVA',
    #                   7.258064516,
    #                   9]
    data = pd.read_excel(filename)
    data['Region'] = data['Unnamed: 0']
    data.drop(labels=labels_to_drop,
              axis='columns',
              inplace=True)
    data.set_index('Region', inplace=True)
    mice = data.columns.to_frame()
    mice.insert(0, 'group', ['naive_male']*4 + 
                            ['naive_female']*4 + 
                            ['trained_male']*3 + 
                            ['trained_female']*3
               )
    data.columns = pd.MultiIndex.from_frame(mice)
    return data

def load_data_one_group(
        filename=paper_folder+'Fall 2021 Compiled Data 10_9_21.xlsx',
        labels_to_drop=['Unnamed: 0']):
    '''
    Loads xlsx spreadsheet of c-fos positive cell counts

    Notes
    -----
    Loads all animals into one group called 'all'
    Currently hardcoded to n=14 animals

    Parameters
    ----------
    filename (string) : Full path to xlsx file
    labels_to_drop (List(string))) : Columns to drop.
                                     Defaults sets first column as 'Region' label, 
                                     then drops first column from main data

    Returns
    -------
    data (pandas.DataFrame) : Pandas Dataframe of c-fos count data
                              Columns are MultiIndex with 2 levels:
                              'group', and mouse name
                              Rows are labelled with 'Region'
                              Access all data for a group with e.g. data['all']
    '''
    # labels_to_drop for original dataset: 
    # ['Unnamed: 0',  
    #                   'Unnamed: 15',
    #                   'Unnamed: 17',
    #                   'Unnamed: 18',
    #                   'interaction in two-way ANOVA',
    #                   7.258064516,
    #                   9]
    data = pd.read_excel(filename)
    data['Region'] = data['Unnamed: 0']
    data.drop(labels=labels_to_drop,
              axis='columns',
              inplace=True)
    data.set_index('Region', inplace=True)
    mice = data.columns.to_frame()
    mice.insert(0, 'group', ['all']*14)
    data.columns = pd.MultiIndex.from_frame(mice)
    return data

def load_brain_regions(
        filename=paper_folder+'Fall 2021 Compiled Data 10_9_21.xlsx',
        sheetname='Functional Organization of Regi'):
    '''
    Loads Brain Region information from xlsx spreadsheet

    Notes
    -----
    in sheetname, brain regions are grouped by anatomical location

    Parameters
    ----------
    filename (string) : Full path to xlsx file
    sheetname (string) : Name of sheet in xlsx file contained region information

    Returns
    -------
    regions (pandas.DataFrame) : Pandas Dataframe of holding brain regino information
    '''
    regions = pd.read_excel(io=filename, sheet_name=sheetname)
    regions.columns=['Cluster','Region','Minor','Major','NA']
    regions.drop(columns=['Minor','Major','NA'], inplace=True)
    regions.sort_values(by='Region', axis=0, key=lambda col: col.str.lower(), inplace=True)
    regions.set_index('Region', inplace=True)
    return regions

def load_brain_regionsB(
        filename=paper_folder+'Fall 2021 Compiled Data 10_9_21.xlsx',
        sheetname='Functional Organization of Regi'):
    '''
    Loads Brain Region information from xlsx spreadsheet

    Notes
    -----
    in sheetname, brain regions are grouped by anatomical location
    only difference with 'load_brain_regions' is that a cluster number is included
    so that regions can be easily sorted exactly as they are in the spreadsheet

    Parameters
    ----------
    filename (string) : Full path to xlsx file
    sheetname (string) : Name of sheet in xlsx file contained region information

    Returns
    -------
    regions (pandas.DataFrame) : Pandas Dataframe of holding brain regino information
    '''
    regions = pd.read_excel(io=filename, sheet_name=sheetname)
    regions.columns=['Cluster','Region','Minor','Major','NA']
    regions.drop(columns=['Minor','Major','NA'], inplace=True)
    regions['Cluster'] = regions.index
    regions.sort_values(by='Region', axis=0, key=lambda col: col.str.lower(), inplace=True)
    regions.set_index('Region', inplace=True)
    return regions

def get_cross_correlation_matrix(data, group='naive_male', method='pearson'):
    '''
    Compute a cross correlation matrix

    Uses pandas.DataFrame method 'corr' to compute the cross correlation matrix
    for a single group within the data. Also extracts matrix in numpy array
    format, with diagonal zeroed, and NaN values zeroed

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                      See pandas.dataframe.corr 

    Returns
    -------
    corrs_df (pandas.DataFrame) : Pandas Dataframe of correlation matrix
                                  Rows and columns are labelled by brain region
    corrs_np (numpy.array) : Numpy array of correlation matrix
    '''
    corrs_df = data[group].transpose().corr(method=method)
    corrs_np = corrs_df.to_numpy()
    np.fill_diagonal(corrs_np, 0)
    corrs_np[np.where(np.isnan(corrs_np))] = 0
    return corrs_df, corrs_np

def get_cross_correlation_pvalues(data, group='naive_male', method='pearson'):
    '''
    Compute p-values for the correlation matrix

    Uses pandas.DataFrame method 'corr' with scipy.stats functions for different
    correlation types, that return p-values. Also extracts matrix in numpy array
    format, NaNs replaced with p-value of 1.

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Selects the scipy function to use for p calculation

    Returns
    -------
    corrs_pvalues_df (pandas.DataFrame) : Pandas Dataframe for matrix of pvalues
                                  Rows and columns are labelled by brain region
    corrs_pvalues_np (numpy.array) : Numpy array of pvalues matrix
    '''
    if method == 'pearson':
        p_func = scipy.stats.pearsonr
    elif method == 'kendall':
        p_func = scipy.stats.kendalltau
    elif method == 'spearmanr':
        p_func = scipy.stats.spearmanr
    else:
        raise ValueError("method must be one of 'pearson', 'kendall', 'spearman'")

    corrs_pvalues_df = data[group].transpose().corr(
        method=lambda x, y: p_func(x, y)[1]
    )
    corrs_pvalues_np = corrs_pvalues_df.to_numpy()
    corrs_pvalues_np[np.where(np.isnan(corrs_pvalues_np))] = 1
    return corrs_pvalues_df, corrs_pvalues_np

def get_adjacency_matrix(data, group='naive_male', 
                         r_threshold=None, p_threshold=0.05, method='pearson'):
    '''
    Find the adjacency matrix for a group

    Uses either the correlation matrix, or p-values matrix, with a threshold
    for either correlation cutoff (greater than a certain correlation coefficient)
    of p-value (lower than a certain p-value). p-values only considered for 
    positive correlations.

    Uses Brain Connectivity Toolbox functions 'bct.utils.threshold_absolute'
    and 'bct.utils.binarize'

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    r_threshold (float) : if r_threshold is not None, find adjacencies based on 
                          correlation coefficients greater than r_threshold
    p_threshold (float) : if r_threshold is None, find adjacencies based on 
                          significance lower than p_threshold
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_cross_correlation_matrix and 
                      get_cross_correlation_pvalues

    Returns
    -------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
    '''
    corr_df, corr_np = get_cross_correlation_matrix(data, group, method=method)
    corr_p_df, corr_p_np = get_cross_correlation_pvalues(data, group, method=method)
    if r_threshold:
        bin_mat = bct.utils.threshold_absolute(corr_np, r_threshold)
        adj_mat = bct.utils.binarize(bin_mat)
    elif p_threshold:
        # replace p values for negative correlations with 1.
        corr_p_np = np.where(corr_np<0, 1.0, corr_p_np)
        bin_mat = bct.utils.threshold_absolute(1-corr_p_np, 1-p_threshold)
        adj_mat = bct.utils.binarize(bin_mat)
    else:
        adj_mat = bct.utils.binarize(corr_np)

    return adj_mat

def adjacency_matrix_metrics(adj_mat):
    '''
    Compute a number of network-level metrics on the adjacency matrix

    Uses Brain Connectivity Toolbox functions for :
        density -- 'bct.density_und'
        transitivity -- 'bct.transitivity_bu'
        efficiency -- 'bct.efficiency_bin'
        community q -- 'bct.community_louvain'
        modularity metric -- 'bct.modularity_und'
        assortativity -- 'bct.assortativity_bin'
        characteristic path length -- 'bct.charpath'
    
    And metrics from this file:
        scale free topology index -- scale_free_topology_index()
        small-worldness -- small_worldness()


    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    metrics (dict) : Dictionary of calculated metric values 
    '''
    metrics = {}
    metrics['density'], metrics['vertices'], metrics['edges'] = bct.density_und(adj_mat)
    metrics['transitivity'] = bct.transitivity_bu(adj_mat)        
    metrics['efficiency'] = bct.efficiency_bin(adj_mat)
    metrics['community_q'] = bct.community_louvain(adj_mat)[1]
    metrics['modularity_metric'] = bct.modularity_und(adj_mat)[1]
    metrics['assortativity'] = bct.assortativity_bin(adj_mat)
    metrics['characteristic_path_length'] = bct.charpath(
        bct.distance_bin(adj_mat),
        include_infinite=False
    )[0]
    metrics['scale_free_topology_index'] = scale_free_topology_index(adj_mat)
    metrics['small_worldness'] = small_worldness(adj_mat)
    return metrics

def adjacency_matrix_nodes_metrics(adj_mat, nr_steps=2):
    '''
    Compute a number of node-level metrics on the adjacency matrix

    Uses Brain Connectivity Toolbox functions for :
        degree -- 'bct.degrees_und'
        topological overlap -- 'bct.gtom'
        matching index -- 'bct.mathing_ind_und'
        clustering coefficient -- 'bct.clustering_coef_bu'
        components, component sizes -- 'bct.get_components'
        community -- 'bct.community_louvain'
        modularity -- 'bct.modularity_und'
        betweenness -- 'bct.betweenness_bin'

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    nr_strps (int) : number of steps used in the bct.gtom function

    Returns
    -------
    metrics (dict) : Dictionary of calculated metric values 
                     Each entry in metrics is a list, order matches the row
                     order of the input adj_mat
    '''
    metrics = {}
    metrics['degree'] = bct.degrees_und(adj_mat)
    metrics['topological_overlap'] = bct.gtom(adj=adj_mat, nr_steps=nr_steps)
    metrics['matching_index'] = bct.matching_ind_und(adj_mat)
    metrics['clustering_coefficient'] = bct.clustering_coef_bu(adj_mat)
    metrics['components'], metrics['component_sizes'] = bct.get_components(adj_mat)
    metrics['community'] = bct.community_louvain(adj_mat)[0]
    metrics['modularity'] = bct.modularity_und(adj_mat)[0]
    metrics['betweenness'] = bct.betweenness_bin(adj_mat)
    return metrics

def get_adj_mats_for_multiple_thresholds(data, group='naive_male', 
                                         threshold_type='p',
                                         thresholds=[0.01, 0.02, 0.03, 0.04, 0.05],
                                         method='pearson'):
    '''
    Find the adjacency matrix for a group for multiple thresholds at once

    Loops through multiple threshold value and calculates a seperate 
    adjacency matrix for each one, and puts them in a dictionary

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    thresholds (list(float)) : threshold values to compute adj_mats for
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_adjacency_matrix

    Returns
    -------
    adj_mat_dict (dict{threshold: numpy.array}) : Dictionary of binary matrices
                                                  of super-threshold correlations
                                                  for each threshold
    '''
    adj_mats_dict = {}
    for threshold in thresholds:
        if threshold_type == 'r':
            adj_mat = get_adjacency_matrix(data=data, group=group, 
                                           r_threshold=threshold, method=method)
        elif threshold_type == 'p':
            adj_mat = get_adjacency_matrix(data=data, group=group, 
                                           p_threshold=threshold, method=method)
        else:
            raise Exception('Unkown threshold_type - should be p or r')

        adj_mats_dict[threshold] = adj_mat

    return adj_mats_dict

def calc_metrics_for_randomizations(adj_mat, num_rands=1000, alpha=1.0, 
                                    extra_metrics={}):
    '''
    Calculate network-level metrics for random networks matched to an 
    adjacency matrix

    Uses Brain Connectivity Toolbox function 'bct.randomizer_bin_und'
    to calculate random graphs from adj_mat, and calculates all network-level
    metrics specified in adjacency_matrix_metrics.
    Then updates the metrics dictionary with any additional, pre-computed 
    metrics provided.

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()
    num_rands (int) : Number of random matrices to generate
    alpha (float) : Fraction of connections to randomize, in 'bct.randomizer_bin_und'
    extra_metrics (dict) : metrics to append to the dictionary of computed
                           metrics

    Returns
    -------
    metrics_list (list) : List of metrics dictionaries, one entry for each
                          random network generated
    '''
    metrics_list = []
    for i in range(num_rands):
        adj_mat_rand = bct.randomizer_bin_und(adj_mat, alpha=alpha)
        metrics = adjacency_matrix_metrics(adj_mat_rand)
        metrics.update(extra_metrics)
        metrics['type'] = 'rand'
        metrics_list.append(metrics)

    return metrics_list

def calc_metrics_for_multiple_thresholds_with_randomizations(
                                    data, 
                                    group='naive_male',
                                    threshold_type='p',
                                    thresholds=[0.01, 0.02,  0.03, 0.04, 0.05],
                                    method='pearson',
                                    num_rands=0,
                                    alpha=1.0):
    '''
    Calculate network-level metrics for a group for multiple thresholds at once,
    with randomizations 

    Basically a helper function to combine get_adj_mats_for_multiple_thresholds 
    and calc_metrics_for_randomizations

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    thresholds (list(float)) : threshold values to compute adj_mats for
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_adjacency_matrix
    num_rands (int) : Number of random matrices to generate
    alpha (float) : Fraction of connections to randomize, in 'bct.randomizer_bin_und'

    Returns
    -------
    metrics_list (list) : List of metrics dictionaries, one entry for each
                          random network generated. 
                          'type': 'real' in dictionary for real data
                          'type': 'rand' in dictionary for random networks
    '''
    adj_mats_dict = get_adj_mats_for_multiple_thresholds(
        data=data,
        group=group,
        threshold_type=threshold_type,
        thresholds=thresholds,
        method=method
    )
    metrics_list = []
    for threshold, adj_mat in adj_mats_dict.items():
        metrics = adjacency_matrix_metrics(adj_mat)
        metrics[threshold_type] = threshold
        metrics['group'] = group
        metrics['type'] = 'real'
        metrics_list.append(metrics)

        rand_nets_metrics_list = calc_metrics_for_randomizations(
            adj_mat=adj_mat,
            num_rands=num_rands,
            alpha=alpha,
            extra_metrics={'group': group, 'type': 'rand', threshold_type: threshold}
        )
        metrics_list.extend(rand_nets_metrics_list)

    return metrics_list

def small_worldness(adj_mat):
    '''
    Calculate small-worldness for a given network

    See Humphries & Gurney (2008):
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002051

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    S_delta_g (float) : Small-worlness metric value    
    '''
    # rand_adj_mat = bct.randmio_und(adj_mat, itr=1)[0]
    # rand_adj_mat = bct.makerandCIJ_und(n=adj_mat.shape[0], k=int(adj_mat.sum()/2))
    rand_adj_mat = nx.convert_matrix.to_numpy_matrix(
        nx.generators.gnm_random_graph(
            n=adj_mat.shape[0], 
            m=int(adj_mat.sum()/2)
        )
    )
    C_delta = bct.transitivity_bu(adj_mat)
    C_delta_rand = bct.transitivity_bu(rand_adj_mat)
    L = bct.charpath(bct.distance_bin(adj_mat), include_infinite=False)[0]
    L_rand = bct.charpath(bct.distance_bin(rand_adj_mat), include_infinite=False)[0]

    # Avoid divide by zero error
    # gamma_delta_g = C_delta/C_delta_rand if C_delta_rand>0.0 else C_delta/1e-9
    # lambda_g = L/L_rand if L_rand>0.0 else L/1e-9
    gamma_delta_g = C_delta/C_delta_rand
    lambda_g = L/L_rand

    S_delta_g = gamma_delta_g/lambda_g
    S_delta_g = np.nan if np.isinf(S_delta_g) else S_delta_g
    return S_delta_g

def copy_upper_triangle(adj_mat):
    '''
    Copy upper trianglular adjacency matrix into lower
    
    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations

    Returns
    -------
    adj_mat_lower_triangle (numpy.array) : Binary matrix of 
                                           super-threshold correlations
    '''
    adj_mat_lower_triangle = adj_mat + adj_mat.T - np.diag(np.diag(adj_mat))
    return adj_mat_lower_triangle

def scale_free_topology_index(adj_mat):
    '''
    Scale-free topology index.
    
    Basically R^2 score for log-scaled degree vs log-scaled density of degree
       See: https://pdfs.semanticscholar.org/75da/d533fb111ba8ca278967b03f137fc2ff6e8e.pdf

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    sft_index (float) : Scale-free topology index metric    
    '''

    degrees = bct.degrees_und(adj_mat)
    # Use histogram of degrees to find densities:
    kde = KernelDensity(kernel='gaussian').fit(degrees.reshape(-1,1))
    degrees_log_dens = kde.score_samples(degrees.reshape(-1,1))
    degrees_dens = np.exp(degrees_log_dens)

    degrees_log10 = np.log10(degrees)
    degrees_log10 = np.where(np.isnan(degrees_log10), 0, degrees_log10)
    degrees_log10 = np.where(np.isinf(degrees_log10), 0, degrees_log10)

    degrees_dens_log10 = np.log10(degrees_dens)
    degrees_dens_log10 = np.where(np.isnan(degrees_dens_log10), 0, degrees_dens_log10)
    degrees_dens_log10 = np.where(np.isinf(degrees_dens_log10), 0, degrees_dens_log10)

    # # Use R^2
    sft_index = r2_score(degrees_log10, degrees_dens_log10)
    # # Instead of R^2 just use linear regression slope
    # regr = linear_model.LinearRegression()
    # regr.fit(degrees_log10.reshape(-1,1), degrees_dens_log10.reshape(-1,1))
    # sft_index = regr.coef_
    return sft_index

def get_adj_mat_dict_for_groups(data, groups, threshold_type='p', 
                                threshold=0.05, method='pearson'):  
    '''
    Calculate adjacency matrix for multiple groups, and put into a dictionary

    Simple wrapper for get_adjacency_matrix

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    groups (list(string)) : List of group names, must be in first MultiIndex level 
                            in data
    r_threshold (float) : if r_threshold is not None, find adjacencies based on 
                          correlation coefficients greater than r_threshold
    p_threshold (float) : if r_threshold is None, find adjacencies based on 
                          significance lower than p_threshold
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_cross_correlation_matrix and 
                      get_cross_correlation_pvalues

    Returns
    -------
    adj_mat_dict (dict{'group': numpy.array}) : Dictionary with group name as key, 
                                                and binary matrix of 
                                                super-threshold correlations
    '''    
    adj_mat_dict = {}
    for group in groups:
        if threshold_type == 'p':
            adj_mat_dict[group] = get_adjacency_matrix(data, group, p_threshold=threshold, method=method)
        elif threshold_type == 'r':
            adj_mat_dict[group] = get_adjacency_matrix(data, group, r_threshold=threshold, method=method)
    return adj_mat_dict

def get_corr_df_dict_for_groups(data, groups, method='pearson'):
    '''
    Calculate cross correlation matrix DataFames for multiple groups, 
    and put into a dictionary

    Simple wrapper for get_cross_correlation_matrix

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    groups (list(string)) : List of group names, must be in first MultiIndex level 
                            in data
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                      Passed to get_cross_correlation_matrix

    Returns
    -------
    corr_df_dict (dict{'group': pandas.DataFrame}) : 
                                Dictionary with group name as key, 
                                and correlation matrix in dataframe form as value
    '''    
    corr_df_dict = {}
    for group in groups:
        corr_df_dict[group], corr_np = get_cross_correlation_matrix(data, group, method=method)
    return corr_df_dict
