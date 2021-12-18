from ast import fix_missing_locations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import netgraph
from matplotlib.pylab import cm
import matplotlib.patches
import json

import bct
# from fa2 import ForceAtlas2
# import For

import funcs
import hubs
import clusters as clust
import figures as figs

mm = 1/25.4 # millimeters in inches

def create_graph(adj_mat):
    '''
    Construct a networkx graph from an adjacency matrix

    Parameters
    ----------
    adj_mat (numpy.array) : Adjacency matrix found with funcs.get_adjacency_matrix

    Returns
    -------
    gr (networkx.Graph) : Networkx Graph object computed from adj_mat
    '''
    rows, cols = np.where(adj_mat==1)
    edges = zip(rows.tolist(), cols.tolist())
    
    all_rows = range(0, adj_mat.shape[0])
    
    gr = nx.Graph()
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    return gr

### NOTE: ForceAtlas2 doesn't seem to work with some versions of python,
### so removed the option to use it here for simplicity of using the scripts

# def calc_positions_forceatlas2(graph):
#     '''
#     Use the ForceAtlas2 package to calculate initial 2D positions for 
#     nodes in a graph, for plotting the graph

#     TODO: allow ForceAtlas2 parameters to be passed as kwargs, or computed 
#     using some method - currently they are fixed based on trial and error

#     Parameters
#     -------
#     gr (networkx.Graph) : Networkx Graph object

#     Returns
#     -------
#     positions_norm (dict) : Dictionary of tuples showing 2D positions for each node
#                             {Region index: (x, y)}
#     '''
#     forceatlas2 = ForceAtlas2(
#         outboundAttractionDistribution=False,
#         scalingRatio=10.0,
#         edgeWeightInfluence=1.0,
#         jitterTolerance=0.2,
#         gravity=40.0,
#         strongGravityMode=False)
#     positions = forceatlas2.forceatlas2_networkx_layout(
#         graph, 
#         pos=None, 
#         iterations=2000)
#     positions_df = pd.DataFrame.from_dict(positions, orient='index', columns=['x', 'y'])
#     positions_norm_df = positions_df.apply(lambda x: (x-x.min())/(x.max()-x.min()))
#     positions_norm = dict([(t.Index, (t.x, t.y)) for t in positions_norm_df.itertuples()])
#     return positions_norm

def save_node_positions(filename, netgraphs_list, groups_list):
    '''
    Put node positions from a list of 'netgraph' objects into a dictionary
    of group: node positions. 

    Used for looping through the groups plotted in the 4-groups graph type figures
    and saving the locations - especially useful after manually editing the 
    positions of the graph nodes interactively.

    Parameters
    ----------
    filename (string) : path to file to save positions dictionary to
    netgraphs_list (list(NetGraph)) : list of objects from the NetGraphs package
    groups_list (list(string)) : list of group names to put into the save dict

    Returns
    -------
    node_pos (dict{group: dict{node_ind: (x,y)}}) : Dictionary, one entry per
                group in groups_list / netgraph in netgraphs_list
                Each entry is a dictionary of node_index : (x, y) tuples
                showing 2D position of each node in the graph
    '''
    node_pos = {}
    for ii, ng in enumerate(netgraphs_list):
        group = groups_list[ii]
        node_pos[group] = ng.node_positions
        node_pos[group] = dict((k, tuple(v)) for k, v in node_pos[group].items())

    json.dump(node_pos, open(filename, 'w'))
    return node_pos

def load_node_positions(filename):
    '''
    Load a file of Put containing a dictionary of group: node positions. 

    Parameters
    ----------
    filename (string) : path to file to load positions dictionary from

    Returns
    -------
    node_pos (dict{group: dict{node_ind: (x,y)}}) : Dictionary, one entry per
                group in groups_list / netgraph in netgraphs_list
                Each entry is a dictionary of node_index : (x, y) tuples
                showing 2D position of each node in the graph
    '''
    node_pos = json.load(open(filename))
    for k, v in node_pos.items():
        node_pos[k] = dict((int(kk), tuple(vv)) for kk, vv in v.items())
    return node_pos

def plot_network_graph_with_clusters(ax, data, adj_mat, clusters, 
                                     cmap=cm.tab20, node_size=50, graph_pos=None):
    '''
    Plotting function. Starting from an adjacency matrix, create a networkx 
    graph to plot, and plot with colors indicating cluster membership

    Parameters
    ----------
    ax (matplotlib.pyplot.axis) : axis object to plot the graph onto
    data (pandas.DataFrame) : data loaded from funcs.load_data function
    adj_mat (numpy.array) : adjacency matrix calculated with 
                            funcs.get_adjacency_matrix function
    clusters (numpy.array) : array of cluster index for each node in the graph
    cmap (matplotlib.cm) : color map for node colors
    node_size (float) : node size for graph nodes
    graph_pos (list(tuples)) : list of positions for each node in the graph
    '''
    gr = create_graph(adj_mat)
    
    all_rows = range(0, adj_mat.shape[0])
    node_labels = dict(zip(all_rows, data.index.to_list()))
    node_sizes = (np.sum(adj_mat, axis=0)*node_size)+node_size

    if graph_pos == None:
        graph_pos = nx.spring_layout(gr, k=0.7)
    
    # if np.any(clusters) == None:
    #     clusters = bct.community_louvain(adj_mat)[0]

    nx.draw_networkx_nodes(
        gr,
        ax=ax,
        pos=graph_pos, 
        node_size=node_sizes,
        node_color=clusters,
        cmap=cmap
    )
    nx.draw_networkx_edges(
        gr,
        ax=ax,
        pos=graph_pos,
        alpha=0.5,
        connectionstyle="arc3,rad=0.2"
    )
    nx.draw_networkx_labels(
        gr,
        ax=ax,
        pos=graph_pos,
        labels=node_labels,
        font_size=10,
    )

def plot_circular_graph_with_clusters(ax, data, adj_mat, clusters, 
                                     cmap=cm.tab20, node_size=50):
    '''
    Plotting function. Starting from an adjacency matrix, create a networkx 
    graph to plot, and plot with colors indicating cluster membership
    using circular layout

    Parameters
    ----------
    ax (matplotlib.pyplot.axis) : axis object to plot the graph onto
    data (pandas.DataFrame) : data loaded from funcs.load_data function
    adj_mat (numpy.array) : adjacency matrix calculated with 
                            funcs.get_adjacency_matrix function
    clusters (numpy.array) : array of cluster index for each node in the graph
    cmap (matplotlib.cm) : color map for node colors
    node_size (float) : node size for graph nodes
    '''
    gr = create_graph(adj_mat)
    
    all_rows = range(0, adj_mat.shape[0])
    node_labels = dict(zip(all_rows, data.index.to_list()))
    node_sizes = (np.sum(adj_mat, axis=0)*node_size)+node_size

    nx.draw_circular(
        gr,
        ax=ax,
        node_size=0,
        node_color=clusters,
        cmap=cmap,
        # labels=node_labels,
        connectionstyle='arc3,rad=1.0'
    )

def make_circular_group_clusters_figure(cmap=cm.tab20,
                                        threshold_type='p',
                                        threshold=0.05,
                                        method='pearson',
                                        cluster_group=None):
    '''
    Figure creation function. 
    
    Process:
    Load data from xlsx.
    Calculate adjacency matrix and cluster memberships for each group
    (Method depends on 'cluster_group' parameter:
        'all' -- compute clusters based on measures from a graph of data across
                 all groups
        'regions' -- cluster order is provided in the xlsx file, extracted
                     using funcs.load_brain_regions()
        group_name -- if 'cluster_group' matches a group name, use cluster IDs
                      based on clustering that named group when plotting all
                      other groups)
    Make a 2x2 subplot of the 4 groups, and use 'plot_circular_graph' to plot
    the networks

    Parameters
    ----------
    cmap (matplotlib.cm) : color map for node colors
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold value to compute adj_mats for
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_adjacency_matrix
    cluster_group (string) : which approach to use for clustering (see above)

    Returns
    -------
    fig (matplotlib.pyplot.figure) : Handle to the generated figure object
    '''
    # Select cluster ordering
    data = funcs.load_data()
    groups = data.columns.get_level_values('group').unique()
    cluster_ids_dict = {}

    if cluster_group == 'all':
        data_all = funcs.load_data_one_group()
        if threshold_type == 'p':
            adj_mat = funcs.get_adjacency_matrix(data_all, group=cluster_group, 
                         r_threshold=None, p_threshold=threshold, method=method)
        elif threshold_type == 'r':
            adj_mat = funcs.get_adjacency_matrix(data_all, group=cluster_group, 
                         r_threshold=threshold, p_threshold=None, method=method)
        clusters = clust.get_cluster_ids(adj_mat)
        for group in groups:
            cluster_ids_dict[group] = clusters

    elif cluster_group == 'regions':
        regions = funcs.load_brain_regions()
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

    adj_mat_dict = funcs.get_adj_mat_dict_for_groups(data, groups,
                                                    threshold_type=threshold_type,
                                                    threshold=threshold, method=method)    

    # Make the figure
    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(2, 2)

    # 4 groups, 4 subplots
    for ind in range(4):
        ax = fig.add_subplot(gs[ind])
        indices = np.argsort(cluster_ids_dict[groups[ind]])
        adj_mat_sorted = adj_mat_dict[groups[ind]][indices][:,indices]
        data_sorted = data.iloc[indices]
        clusters_sorted = cluster_ids_dict[groups[ind]][indices]
        g = plot_circular_graph_with_clusters(ax=ax, 
                                              data=data_sorted,
                                              adj_mat=adj_mat_sorted,
                                              clusters=clusters_sorted,
                                              cmap=cmap)
        ax.set_title(groups[ind])

    return fig

def make_centrality_measures_two_panel_plot(data, group='naive_male',
                              threshold_type='p', threshold=0.05, method='pearson',
                              cutoff=0.2):
    '''
    Figure creation function. 
    
    Process:
    Load data from xlsx.
    Calculate adjacency matrix and cluster memberships for each group
    (Method depends on 'cluster_group' parameter:
        'all' -- compute clusters based on measures from a graph of data across
                 all groups
        'regions' -- cluster order is provided in the xlsx file, extracted
                     using funcs.load_brain_regions()
        group_name -- if 'cluster_group' matches a group name, use cluster IDs
                      based on clustering that named group when plotting all
                      other groups)
    Make a 2x2 subplot of the 4 groups, and use 'plot_circular_graph' to plot
    the networks

    Parameters
    ----------
    cmap (matplotlib.cm) : color map for node colors
    threshold_type (string) : {'p', 'r'} -- determines whether the thresholds
                              should be passed to 'get_adjacency_matrix' as 
                              'p_threshold' or 'r_threshold'
    threshold (float) : threshold value to compute adj_mats for
    method (string) : {‘pearson’, ‘kendall’, ‘spearman’}
                      Passed to both get_adjacency_matrix
    cluster_group (string) : which approach to use for clustering (see above)

    Returns
    -------
    fig (matplotlib.pyplot.figure) : Handle to the generated figure object
    '''
    top_betweenness_to_plot_df, top_degree_to_plot_df = hubs.centrality_measures_with_hub_regions(
                                            data=data, group=group, threshold_type=threshold_type, 
                                            threshold=threshold,  method=method, cutoff=cutoff)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1)

    ax = fig.add_subplot(gs[0])
    g = sns.barplot(
        data=top_betweenness_to_plot_df, 
        x=top_betweenness_to_plot_df.index, 
        y='betweenness',
        hue='hub_region', dodge=False, palette={True: 'r', False: 'k'},
        ax=ax
    )
    ax.legend_ = None
    plt.xlabel(None)
    plt.ylabel('Betweenness')
    plt.xticks(rotation=90)
    plt.axvline(x=top_betweenness_to_plot_df.shape[0]//2-.5, ls='--', c='k', lw=2)
    plt.box(on=False)
    

    ax = fig.add_subplot(gs[1])
    g = sns.barplot(
        data=top_degree_to_plot_df, 
        x=top_degree_to_plot_df.index, 
        y='degree',
        hue='hub_region', dodge=False, palette={True: 'r', False: 'k'},
        ax=ax
    )
    ax.legend_ = None
    plt.xlabel(None)
    plt.ylabel('Degree')
    plt.xticks(rotation=90)
    plt.axvline(x=top_betweenness_to_plot_df.shape[0]//2-.5, ls='--', c='k', lw=2)
    plt.box(on=False)

    plt.tight_layout(pad=0.8)

    return fig

def plot_bar_betweenness(ax, centrality_data, cutoff, bar_colors=['r', 'k'], y_upper_lim=None):
    '''
    Plotting function. 
    
    Bar plot of betweenness values for regions. Adds a 'cutoff' line showing some
    portion of the regions as above a cutoff for being considered a hub region

    Parameters
    ----------
    ax (matplotlib.pyplot.axis) : axis object to plot the graph onto
    centrality_data (pandas.DataFrame) : data from hubs.centrality_df
    cutoff (float) : Fraction of nodes to include in the bar plot
    bar_colors (List(colors)) : List of colors, index 0 is for hubs, index 1 for non-hubs
    y_upper_lim (float) : Optional, limit of the y-axis, used for setting y-limits the 
                          same across multiple plots

    Returns
    -------
    g (sns.barplot) : Handle to the generated plot object
    '''
    top_betweenness_to_plot_df = hubs.get_top_nodes_by_betweenness(
        centrality_df=centrality_data,
        cutoff=cutoff)
    g = sns.barplot(
        data=top_betweenness_to_plot_df, 
        x=top_betweenness_to_plot_df.index, 
        y='betweenness',
        hue='hub_region', dodge=False, palette={True: bar_colors[0], False: bar_colors[1]},
        ax=ax
    )
    ax.legend_ = None
    plt.xlabel(None)
    plt.ylabel('Betweenness')
    if y_upper_lim:
        plt.ylim([0, y_upper_lim])
    plt.xticks(rotation=90)
    plt.axvline(x=top_betweenness_to_plot_df.shape[0]//3*2-.5, ls=':', c='0.2', lw=0.8)
    plt.box(on=False)
    return g

def plot_bar_degree(ax, centrality_data, cutoff, bar_colors=['r', 'k'], y_upper_lim=None):
    '''
    Plotting function. 
    
    Bar plot of degree values for regions. Adds a 'cutoff' line showing some
    portion of the regions as above a cutoff for being considered a hub region

    Parameters
    ----------
    ax (matplotlib.pyplot.axis) : axis object to plot the graph onto
    centrality_data (pandas.DataFrame) : data from hubs.centrality_df
    cutoff (float) : Fraction of nodes to include in the bar plot
    bar_colors (List(colors)) : List of colors, index 0 is for hubs, index 1 for non-hubs
    y_upper_lim (float) : Optional, limit of the y-axis, used for setting y-limits the 
                          same across multiple plots

    Returns
    -------
    g (sns.barplot) : Handle to the generated plot object
    '''
    top_degree_to_plot_df = hubs.get_top_nodes_by_degree(
        centrality_df=centrality_data,
        cutoff=cutoff)
    g = sns.barplot(
        data=top_degree_to_plot_df,
        x=top_degree_to_plot_df.index, 
        y='degree',
        hue='hub_region', dodge=False, palette={True: bar_colors[0], False: bar_colors[1]},
        ax=ax
    )
    ax.legend_ = None
    plt.xlabel(None)
    plt.ylabel('Degree')
    if y_upper_lim:
        plt.ylim([0, y_upper_lim])
    plt.xticks(rotation=90)
    plt.axvline(x=top_degree_to_plot_df.shape[0]//3*2-.5, ls=':', c='0.2', lw=0.5)
    plt.box(on=False)
    # plt.tight_layout(pad=1.0)
    return g

def plot_network_graph(ax, hub_data, adj_mat, cluster_ids, 
                       node_layout='community',
                       node_positions=None,
                       edge_layout='straight',
                       node_color_map='tab20',
                       node_alpha=1.0,
                       border_color_map=['k', 'r'],
                       border_width_map=[0.0, 1.0],
                       node_label_fontdict={'size': 6},
                       node_size=3.,
                       edge_alpha=1.,
                       edge_color='k',
                       edge_width=1.):
    '''
    Plotting function. 
    
    Network graph plot.

    Parameters
    ----------
    ax (matplotlib.pyplot.axis) : axis object to plot the graph onto
    centrality_data (pandas.DataFrame) : data from hubs.centrality_df
    cutoff (float) : Fraction of nodes to include in the bar plot
    bar_colors (List(colors)) : List of colors, index 0 is for hubs, index 1 for non-hubs
    y_upper_lim (float) : Optional, limit of the y-axis, used for setting y-limits the 
                          same across multiple plots

    Returns
    -------
    g (sns.barplot) : Handle to the generated plot object
    '''
    gr = create_graph(adj_mat)
    node_inds = range(len(hub_data.index))

    node_to_community = clust.get_node_to_cluster(cluster_ids)
    node_labels = dict(zip(node_inds, hub_data.index))

    if isinstance(node_color_map, str):
        colors = plt.get_cmap(node_color_map).colors
    else:
        colors = node_color_map
    node_color = dict(zip(node_inds, [colors[int(ind)]  for ind in cluster_ids]))

    border_colors = [border_color_map[1] if ishub else border_color_map[0] for 
                                        ishub in hub_data['hub_region']]
    node_edge_color = dict(zip(node_inds, border_colors))

    border_widths = [border_width_map[1] if ishub else border_width_map[0] for 
                                        ishub in hub_data['hub_region']]
    node_edge_width = dict(zip(node_inds, border_widths))

    # if node_layout == 'force':
        # node_layout = calc_positions_forceatlas2(gr)
    if node_layout == 'set_position':
        node_layout = node_positions

    g = netgraph.InteractiveGraph(gr,
                       node_layout=node_layout,
                       node_layout_kwargs={'node_to_community': node_to_community},
                       node_labels=node_labels,
                       node_color=node_color,
                       node_alpha=node_alpha,
                       node_edge_color=node_edge_color,
                       node_edge_width=node_edge_width,
                       node_label_fontdict=node_label_fontdict,
                       node_size=node_size,
                       edge_layout=edge_layout,
                       edge_alpha=edge_alpha,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       ax=ax)

    return g

def make_cluster_graph_figure(hub_data, adj_mat, cluster_ids,
                              cutoff=0.4,
                              figsize=(8, 12)):
    trimmed_data, trimmed_adj_mat, trimmed_cluster_ids = clust.trim_disconnected_nodes(
        data=hub_data,
        adj_mat=adj_mat,
        cluster_ids=cluster_ids,
        min_size=3
    )

    fig = plt.figure(figsize = figsize)
    gs = fig.add_gridspec(2, 1, wspace = 0, hspace = 0.0,
                          height_ratios=[2/3, 1/3])
    gs1 = gs[1].subgridspec(2,1, wspace = 0.0, hspace = 0.0)

    # Graph panel
    ax = fig.add_subplot(gs[0])
    ga = plot_network_graph(ax=ax, 
                           hub_data=trimmed_data, 
                           adj_mat=trimmed_adj_mat,
                           cluster_ids=trimmed_cluster_ids)

    # Betweeenness panel
    ax = fig.add_subplot(gs1[0])
    gb = plot_bar_betweenness(ax=ax,
                             centrality_data=hub_data,
                             cutoff=cutoff)

    # Degree panel
    ax = fig.add_subplot(gs1[1])
    gc = plot_bar_degree(ax=ax,
                         centrality_data=hub_data,
                         cutoff=cutoff)

    plt.tight_layout(pad=1.5)

    return fig, ga, gb, gc

def make_cluster_graph_4groups_figure(hub_data_dict, adj_mat_dict, cluster_ids_dict,
                                      groups=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
                                      group_colors={'naive_male': 'r', 'naive_female': 'r', 'trained_male': 'r', 'trained_female': 'r'},
                                      node_color_map='tab20',
                                      cutoff=0.3,
                                      node_layout='community',
                                      node_positions=None,
                                      node_position_file=None,
                                      fontsize=8,
                                      figsize=(178*mm, 237*mm), filename='clust_gr_4grp_fig.pdf'):

    # calculate upper y limit for centrality bar plots across groups
    betweenness_all_groups = []
    degree_all_groups = []
    for group in groups:
        betweenness_all_groups.extend(list(hub_data_dict[group]['betweenness']))
        degree_all_groups.extend(list(hub_data_dict[group]['degree']))
    max_betweenness = np.max(betweenness_all_groups)
    max_degree = np.max(degree_all_groups)

    figure_out = {}

    if node_position_file is not None:
        node_positions = load_node_positions(node_position_file)

    node_label_size = (fontsize*3/4)
    ticksize = (fontsize*3/4)

    gray_color = '0.3'

    matplotlib.rcParams['font.family'] = ['Arial', 'sans-serif']
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = ticksize
    matplotlib.rcParams['ytick.labelsize'] = ticksize

    fig = plt.figure(figsize=figsize, tight_layout=True)
    gs = fig.add_gridspec(2, 2, wspace = 0.1, hspace = 0.1)

    for ii, group in enumerate(groups):
        trimmed_data, trimmed_adj_mat, trimmed_cluster_ids = clust.trim_disconnected_nodes(
            data=hub_data_dict[group],
            adj_mat=adj_mat_dict[group],
            cluster_ids=cluster_ids_dict[group],
            min_size=3
        )

        gs_ii = gs[ii].subgridspec(2, 1, wspace = 0.0, hspace = 0.0,
                            height_ratios=[3/4, 1/4])
        gs_ii_ab = gs_ii[0].subgridspec(1, 2, wspace=0.0, hspace=0.0,
                                     width_ratios = [1, 0])#[2/3, 1/3])
        gs_ii_cd = gs_ii[1].subgridspec(2,1, wspace = 0.0, hspace = 0.6)

        # Graph panel
        ax = fig.add_subplot(gs_ii_ab[0])
        ga = plot_network_graph(ax=ax,
                                hub_data=trimmed_data, 
                                adj_mat=trimmed_adj_mat,
                                cluster_ids=trimmed_cluster_ids,
                                node_layout=node_layout,
                                node_positions=node_positions[group],
                                node_color_map=node_color_map,
                                node_alpha=1.0,
                                border_color_map=[gray_color, group_colors[group]],
                                border_width_map=[0.4, 1.0],
                                node_label_fontdict={'size': node_label_size},
                                node_size=5.,
                                edge_width=0.2,
                                edge_color=gray_color)

        # Betweeenness panel
        ax = fig.add_subplot(gs_ii_cd[0])
        gb = plot_bar_betweenness(ax=ax,
                                  centrality_data=hub_data_dict[group],
                                  cutoff=cutoff,
                                  bar_colors=[group_colors[group], gray_color],
                                  y_upper_lim=max_betweenness)

        # Degree panel
        ax = fig.add_subplot(gs_ii_cd[1])
        gc = plot_bar_degree(ax=ax,
                             centrality_data=hub_data_dict[group],
                             cutoff=cutoff,
                             bar_colors=[group_colors[group], gray_color],
                             y_upper_lim=max_degree)

        figure_out[group] = fig, ga, gb, gc

    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return figure_out
