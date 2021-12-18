import pandas as pd
import numpy as np

import funcs

def centrality_df(data, group='naive_male',
                          threshold_type='p', threshold=0.05, method='pearson'):
    '''
    Compute degree and betweenness for a data set

    Wrapper function to calculate an adjacency matrix on a group,
    then compute all metrics and keep just degree and betweenness

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold to compute adj_mat
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to get_adjacency_matrix

    Returns
    -------
    centrality_df (pandas.DataFrame) : DataFrame containing just degree and betweenness
    '''
    if threshold_type == 'p':
        adj_mat = funcs.get_adjacency_matrix(data, group=group, p_threshold=threshold, method=method)
    elif threshold_type == 'r':
        adj_mat = funcs.get_adjacency_matrix(data, group=group, r_threshold=threshold, method=method)

    metrics = funcs.adjacency_matrix_nodes_metrics(adj_mat)

    centrality_df = pd.DataFrame(
        [metrics['degree'], metrics['betweenness']],
        index=['degree', 'betweenness'], columns=data.index).transpose()

    return centrality_df

def get_top_nodes_by_degree(centrality_df, cutoff=0.2):
    '''
    Find a fraction of top nodes for the degree measure

    Parameters
    ----------
    centrality_df (pandas.DataFrame) : DataFrame containing just degree and betweenness
                                       Computed with 'centrality_df()'
    cutoff (float) : Fraction of nodes to include in the returned dataframe

    Returns
    -------
    top_degrees_df (pandas.DataFrame) : DataFrame of just the top 'cutoff' fraction of 
                                        nodes from centrality_df
    '''
    cutoff_ind = int(len(centrality_df)*cutoff)
    degree_indices = np.argsort(centrality_df['degree'])
    top_degrees_df = centrality_df.loc[centrality_df.index[np.flipud(degree_indices)]]
    top_degrees_df = top_degrees_df.iloc[:cutoff_ind]
    return top_degrees_df

def get_top_nodes_by_betweenness(centrality_df, cutoff=0.2):
    '''
    Find a fraction of top nodes for the betweenness measure

    Parameters
    ----------
    centrality_df (pandas.DataFrame) : DataFrame containing just degree and betweenness
                                       Computed with 'centrality_df()'
    cutoff (float) : Fraction of nodes to include in the returned dataframe

    Returns
    -------
    top_betweenness_df (pandas.DataFrame) : DataFrame of just the top 'cutoff' fraction of 
                                        nodes from centrality_df
    '''
    cutoff_ind = int(len(centrality_df)*cutoff)
    betweenness_indices = np.argsort(centrality_df['betweenness'])
    top_betweenness_df = centrality_df.loc[centrality_df.index[np.flipud(betweenness_indices)]]
    top_betweenness_df = top_betweenness_df.iloc[:cutoff_ind]
    return top_betweenness_df

def centrality_measures_with_hub_regions(data, group='naive_male',
                              threshold_type='p', threshold=0.05, method='pearson',
                              cutoff=0.2):
    '''
    Determine hub regions based on top 'cutoff' fraction of nodes for 
    degree and betweenness for a data set

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold to compute adj_mat
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to get_adjacency_matrix
    cutoff (float) : Fraction of nodes for centrality measures to determine 
                     hub regions

    Returns
    -------
    hub_df (pandas.DataFrame) : DataFrame containing degree and betweenness    
                                and a boolean for determined hub regions
                                in 'hub_region' column
    '''
    hub_df = centrality_df(data=data, group=group, threshold_type=threshold_type, 
                           threshold=threshold, method=method)
    top_betweenness_df = get_top_nodes_by_betweenness(hub_df, cutoff=cutoff)
    top_degree_df = get_top_nodes_by_degree(hub_df, cutoff=cutoff)
    hub_regions = [x for x in top_betweenness_df.index.to_list() if x in top_degree_df.index.to_list()]
    hub_df['hub_region'] = hub_df.index.isin(hub_regions)
    return hub_df

def centrality_measures_with_hub_regions_to_plot(data, group='naive_male',
                              threshold_type='p', threshold=0.05, method='pearson',
                              cutoff=0.2):
    '''
    Create a DataFrame to be used for making centrality measure plots
    showing 2x cutoff fractino of nodes, with the cutoff for inclusion as
    a hub at 1x cutoff

    Parameters
    ----------
    data (pandas.DataFrame) : data loaded into DataFrame format by 'load_data' function
    group (string) : Group name, must be in first MultiIndex level in data
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold to compute adj_mat
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to get_adjacency_matrix
    cutoff (float) : Fraction of nodes for centrality measures to determine 
                     hub regions

    Returns
    -------
    top_betweenness_to_plot_df (pandas.DataFrame) : DataFrame containing betweenness
                    measures to plot, along with hub_region boolean
    top_degree_to_plot_df (pandas.DataFrame) : DataFrame containing degree
                    measures to plot, along with hub_region boolean
    '''
    hub_df = centrality_df(data=data, group=group, threshold_type=threshold_type, 
                           threshold=threshold, method=method)
    top_betweenness_df = get_top_nodes_by_betweenness(hub_df, cutoff=cutoff)
    top_degree_df = get_top_nodes_by_degree(hub_df, cutoff=cutoff)
    hub_regions = [x for x in top_betweenness_df.index.to_list() if x in top_degree_df.index.to_list()]

    top_degree_to_plot_df = get_top_nodes_by_degree(hub_df, cutoff=cutoff*2)
    top_degree_to_plot_df['hub_region'] = top_degree_to_plot_df.index.isin(hub_regions)
    top_degree_to_plot_df.loc[
        top_degree_to_plot_df.index[top_degree_to_plot_df.shape[0]//2:],'hub_region'] = False

    top_betweenness_to_plot_df = get_top_nodes_by_betweenness(hub_df, cutoff=cutoff*2)
    top_betweenness_to_plot_df['hub_region'] = top_betweenness_to_plot_df.index.isin(hub_regions)
    top_betweenness_to_plot_df.loc[
        top_betweenness_to_plot_df.index[top_betweenness_to_plot_df.shape[0]//2:],'hub_region'] = False

    return top_betweenness_to_plot_df, top_degree_to_plot_df

def hub_counts_for_multiple_trials(hub_dict_list):
    '''
    Take Hub information from multiple trials, provided as a list of dictionaries,
    one entry for each 'trial', and count the number of times a Region is included
    as a Hub across all the trials.

    Parameters
    ----------
    hub_dict_list (List(dict)) : list of dictionaries containing hub information
                                 calculated from centrality_measures_with_hub_regions,
                                 one entry in the list from each 'trial'

    Returns
    -------
    hubs_all_trials_df (pandas.DataFrame) : DataFrame of Counts of number of times
                    that a Region in considered a hub, from all trials provided 
                    in hub_dict_list
    '''
    hubs_all_trials = []
    for hub_dict in hub_dict_list:
        hubs = list(hub_dict.loc[hub_dict['hub_region']==True].index)
        hubs_all_trials.extend(hubs)
    hubs_all_trials_df = pd.DataFrame(np.unique(hubs_all_trials, return_counts=True)).transpose()
    hubs_all_trials_df.columns=['Region', 'count']
    hubs_all_trials_df.set_index('Region')
    return hubs_all_trials_df
