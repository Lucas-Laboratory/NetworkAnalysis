from math import ceil
from networkx.generators.classic import null_graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import seaborn as sns
import networkx as nx
import bct


def plot_network(data, adj_mat, clusters=None, cmap=None):
    plt.figure(figsize=(14,14))
    rows, cols = np.where(adj_mat==1)
    edges = zip(rows.tolist(), cols.tolist())
    
    all_rows = range(0, adj_mat.shape[0])
    node_labels = dict(zip(all_rows, data.index.to_list()))
    node_sizes = (np.sum(adj_mat, axis=0)*50)+50
    
    gr = nx.Graph()
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    
    graph_pos = nx.spring_layout(gr, k=0.7)

    if cmap == None:
        cmap = cm.tab20
    
    if np.any(clusters) == None:
        clusters = bct.community_louvain(adj_mat)[0]

    nx.draw_networkx_nodes(
        gr, 
        pos=graph_pos, 
        node_size=node_sizes,
        node_color=clusters,
        cmap=cmap
    )
    nx.draw_networkx_edges(
        gr,
        pos=graph_pos,
        alpha=0.5,
        connectionstyle="arc3,rad=0.2"
    )
    nx.draw_networkx_labels(
        gr,
        pos=graph_pos,
        labels=node_labels,
        font_size=10,
    )
    plt.show()

def plot_metrics_for_multiple_thresholds(
    metrics_df,
    threshold_type='p',
    metrics_to_plot=['density', 'vertices', 'edges', 'transitivity', 'efficiency',
                     'community_q', 'modularity_metric', 'assortativity',
                     'characteristic_path_length', 'scale_free_topology_index',
                     'small_worldness'],
    num_cols=4, x_scale=4, y_scale=3):

    num_rows = ceil(len(metrics_to_plot)/num_cols)

    fig, axs = plt.subplots(nrows=num_rows,
                            ncols=num_cols,
                            figsize=(num_cols*x_scale, num_rows*y_scale),
                            tight_layout=True)

    for plot_num, metric_name in enumerate(metrics_to_plot):
        ax=axs[plot_num//num_cols, plot_num%num_cols]

        sns.lineplot(data=metrics_df,
                     x=threshold_type,
                     y=metric_name,
                     hue='type',
                     ci='sd',
                     ax=ax)
        # ax.set_ylim(bottom=0.0)

def plot_metrics_for_multiple_groups_for_multiple_thresholds(
    metrics_df,
    plot_type='line',
    threshold_type='p',
    metrics_to_plot=['density', 'vertices', 'edges', 'transitivity', 'efficiency',
                     'community_q', 'modularity_metric', 'assortativity',
                     'characteristic_path_length', 'scale_free_topology_index',
                     'small_worldness'],
    num_cols=4, x_scale=4, y_scale=3):

    num_rows = ceil(len(metrics_to_plot)/num_cols)

    fig, axs = plt.subplots(nrows=num_rows,
                            ncols=num_cols,
                            figsize=((num_cols*x_scale)+2, num_rows*y_scale),
                            tight_layout=True)

    for plot_num, metric_name in enumerate(metrics_to_plot):
        ax=axs[plot_num//num_cols, plot_num%num_cols]

        if plot_type == 'line':
            sns.lineplot(data=metrics_df,
                        x=threshold_type,
                        y=metric_name,
                        hue='group',
                        style='type',
                        ci='sd',
                        ax=ax)
        elif plot_type == 'scatter':
            sns.scatterplot(data=metrics_df,
                            x=threshold_type,
                            y=metric_name,
                            hue='group',
                            style='type',
                            markers=['X', '.'],
                            linewidth=0,
                            alpha=0.5,
                            ax=ax)
        elif plot_type == 'swarm':
            sns.swarmplot(data=metrics_df,
                          x=threshold_type,
                          y=metric_name,
                          hue='group',
                          dodge=True,
                          linewidth=0,
                          size=2,
                          ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=metrics_df,
                        x=threshold_type,
                        y=metric_name,
                        hue='group',
                        linewidth=0.1,
                        fliersize=1,
                        dodge=True,
                        ax=ax)

        # ax.set_ylim(bottom=0.0)
        if plot_num == 0:
            fig.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        ax.legend([], [], frameon=False)

def plot_cross_correlation_matrix(corrs, figsize=(26,22), cmap=None):
    fig = plt.figure(figsize=figsize)
    if cmap == None:
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
    sns.heatmap(corrs_df_sorted, center=0, cmap=cmap)
    return fig

