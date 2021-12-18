import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import funcs
import clusters as clust

mm = 1/25.4 # millimeters in inches

def make_correlation_matrix_group_clusters_figure(
    groups=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    cmap=sns.diverging_palette(220, 10, s=65, l=55, center='light', sep=128, as_cmap=True),
    # cmap=matplotlib.cm.get_cmap('RdGy_r'),
    threshold_type='p', threshold=0.05, method='pearson', cluster_group=None,
    fontsize=8, figsize=(178*mm, 152*mm), filename='corr_mat_4grp_fig.pdf'
):

    # 1) Initial setup
    groups_out = {}

    node_label_size = (fontsize*3/4)
    ticksize = (fontsize*3/4)

    gray_color = '0.3'

    matplotlib.rcParams['font.family'] = ['Arial', 'sans-serif']
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = ticksize
    matplotlib.rcParams['ytick.labelsize'] = ticksize

    # 2) Select cluster ordering
    data = funcs.load_data()
    # groups = data.columns.get_level_values('group').unique()
    cluster_ids_dict = {}
    if cluster_group == 'regions':
        regions = funcs.load_brain_regions()
        clusters = regions['Cluster'].to_numpy()
        for group in groups:
            cluster_ids_dict[group] = clusters
    elif cluster_group == 'regionsB':
        # Order by column B in the spreadsheet
        regions = funcs.load_brain_regionsB()
        clusters = regions['Cluster'].to_numpy()
        for group in groups:
            cluster_ids_dict[group] = clusters
    elif cluster_group in groups:
        if threshold_type == 'p':
            adj_mat = funcs.get_adjacency_matrix(data, group=cluster_group, 
                         r_threshold=None, p_threshold=threshold, method=method)
        elif threshold_type == 'r':
            adj_mat = funcs.get_adjacency_matrix(data, group=cluster_group, 
                         r_threshold=threshold, p_threshold=None, method=method)
        clusters = clust.get_cluster_ids(adj_mat)
        for group in groups:
            cluster_ids_dict[group] = clusters
    elif cluster_group == None:
        cluster_ids_dict = clust.get_cluster_ids_for_groups(data, groups, 
                        threshold_type=threshold_type, threshold=threshold, method=method)
    
    corr_df_dict = funcs.get_corr_df_dict_for_groups(data, groups, method=method)
    
    # 3) Make the figure
    fig = plt.figure(figsize=figsize, tight_layout=True)
    gs = fig.add_gridspec(2, 2, wspace = 0.1, hspace = 0.1)

    # 4 groups, 4 subplots
    for ii, group in enumerate(groups):
        ax = fig.add_subplot(gs[ii])
        corrs_df_sorted = clust.rearrange_corr_df_by_clusters(
            corr_df_dict[group], 
            cluster_ids_dict[group]
        )
        g = sns.heatmap(corrs_df_sorted, 
                        vmin=-1., vmax=1., cmap=cmap, 
                        xticklabels=True, yticklabels=True,
                        ax=ax)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.set_title(group)
        groups_out[group] = g

    # 4) Save the figure
    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return fig, groups_out

def make_correlation_matrix_all_clusters_figure(
    cmap=sns.diverging_palette(150, 275, s=80, l=55, center='light', sep=20, as_cmap=True),
    p_threshold=0.05, method='pearson'):

    # 1) Pull in all data as one group, to get cluster 
    # order based on correlations among all animals
    data_all = funcs.load_data_one_group()
    adj_mat_all = funcs.get_adjacency_matrix(
        data=data_all,
        group='all',
        p_threshold=0.05,
        method=method
    )
    clusters_all_arr = clust.get_cluster_ids(adj_mat_all)

    # 2) Pull in data with individual groups
    # Create a correlation matrix plot for each group
    # with columns/rows sorted by clusters selected above
    data = funcs.load_data()
    groups = data.columns.get_level_values('group').unique()
    corr_df_dict = funcs.get_corr_df_dict_for_groups(data, groups, method=method)

    # 3) Make the figure
    fig = plt.figure(figsize=(22,22))
    gs = fig.add_gridspec(2, 2)

    # 4 groups, 4 subplots
    for ind in range(4):
        ax = fig.add_subplot(gs[ind])
        corrs_df_sorted = clust.rearrange_corr_df_by_clusters(
            corr_df_dict[groups[ind]], 
            clusters_all_arr
        )
        g = sns.heatmap(corrs_df_sorted, center=0, cmap=cmap, ax=ax)
        ax.set_title(groups[ind])

    return fig
