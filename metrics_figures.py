from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import matplotlib
import seaborn as sns

mm = 1/25.4 # millimeters in inches

def plot_multiple_thresholds_metric_line_plot(
    ax, metrics_df, metric_name,
    groups_order=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    groups_colors=['k', 'k', 'r', 'r'],
    groups_mfc=['None', 'k', 'None', 'r'],
    groups_markers_real=['o','o','o','o'],
    groups_markers_rand=['X','X','X','X'],
    groups_lines=['--','-','--','-'],
    threshold_type='p'
):
    for ii, group in enumerate(groups_order):
        group_data = metrics_df.loc[metrics_df['group'] == group]

        # one line for real
        group_data_real = group_data[group_data['type'] == 'real']
        x = group_data_real.groupby(threshold_type).median().index.values
        y = group_data_real.groupby(threshold_type).median()[metric_name].values
        y_lower = group_data_real.groupby(threshold_type).min()[metric_name].values
        y_upper = group_data_real.groupby(threshold_type).max()[metric_name].values
        ax.plot(x, y, groups_lines[ii], color=groups_colors[ii], lw=1)
        ax.plot(x, y, groups_markers_real[ii], mfc=groups_mfc[ii], 
                color=groups_colors[ii], lw=1, mew=0.5, ms=4)
        ax.fill_between(x, y_lower, y_upper, color=groups_colors[ii], 
                        alpha=0.1, lw=1, ls=groups_lines[ii])

        # one line for rand
        group_data_rand = group_data[group_data['type'] == 'rand']
        x = group_data_rand.groupby(threshold_type).median().index.values
        y = group_data_rand.groupby(threshold_type).median()[metric_name].values
        y_lower = group_data_rand.groupby(threshold_type).min()[metric_name].values
        y_upper = group_data_rand.groupby(threshold_type).max()[metric_name].values
        ax.plot(x, y, groups_lines[ii], color=groups_colors[ii], lw=1)
        ax.plot(x, y, groups_markers_rand[ii], mfc=groups_mfc[ii], 
                color=groups_colors[ii], lw=1, mew=0.5, ms=4)
        ax.fill_between(x, y_lower, y_upper, color=groups_colors[ii], 
                        alpha=0.1, lw=1, ls=groups_lines[ii])
    ax.set_ylabel(metric_name)
    ax.set_xlabel('p')
    g = ax
    return g

def plot_one_threshold_metric_strip_plot_mpl(
    ax, metrics_df, metric_name,
    groups_order=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    groups_colors=['k', 'white', 'r', 'white'],
    groups_mfc=['white', 'k', 'white', 'r'],
    groups_markers_real=['o','o','o','o'],
    groups_markers_rand=['X','X','X','X'],
    threshold_type='p', threshold=0.05
):

    # threshold indices are not always precise, so:
    thresholds = metrics_df[threshold_type].unique()
    true_threshold = thresholds[np.abs(thresholds - threshold).argmin()]
    metrics_at_threshold = metrics_df.loc[metrics_df[threshold_type]==true_threshold]

    for ii, group in enumerate(groups_order):
        group_data = metrics_at_threshold.loc[metrics_at_threshold['group'] == group]

        # one set of points for real
        group_data_real = group_data[group_data['type'] == 'real']
        x = np.random.uniform(-0.4, 0.4, group_data_real.shape[0]) + ii
        y = group_data_real[metric_name].values
        ax.plot(x, y, groups_markers_real[ii], mfc=groups_mfc[ii], 
                color=groups_colors[ii], lw=0.5, mew=0.5, ms=4, alpha=1.)

        # one line for rand
        group_data_rand = group_data[group_data['type'] == 'rand']
        x = np.random.uniform(-0.4, 0.4, group_data_rand.shape[0]) + ii
        y = group_data_rand[metric_name].values
        ax.plot(x, y, groups_markers_rand[ii], mfc=groups_mfc[ii], 
                color=groups_colors[ii], lw=0.5, mew=0.5, ms=4, alpha=1.)

    ax.set_ylabel(metric_name)
    g = ax
    return g

def plot_multiple_thresholds_metric_swarm_plot(
    ax, metrics_df, metric_name,
    groups_order=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    groups_colors=['k', 'k', 'r', 'r'],
    threshold_type='p', dodge=True
):
    g = sns.swarmplot(data=metrics_df,
                      x=threshold_type,
                      y=metric_name,
                      hue='group',
                      palette=dict(zip(groups_order, groups_colors)),
                      dodge=dodge,
                      linewidth=0,
                      size=2,
                      ax=ax)
    return g

def plot_one_threshold_metric_strip_plot(
    ax, metrics_df, metric_name,
    groups_order=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    groups_colors=['k', 'k', 'r', 'r'],
    threshold_type='p',
    threshold=0.05, dodge=True
):
    # threshold indices are not always precise, so:
    thresholds = metrics_df[threshold_type].unique()
    true_threshold = thresholds[np.abs(thresholds - threshold).argmin()]
    metrics_at_threshold = metrics_df.loc[metrics_df[threshold_type]==true_threshold]
    metrics_at_threshold_rand = metrics_at_threshold.loc[metrics_at_threshold['type']=='rand']
    metrics_at_threshold_real = metrics_at_threshold.loc[metrics_at_threshold['type']=='real']
    g_rand = sns.stripplot(data=metrics_at_threshold_rand,
                           x=threshold_type,
                           y=metric_name,
                           hue='group',
                           hue_order=groups_order,
                           palette=dict(zip(groups_order, groups_colors)),
                           dodge=dodge,
                           linewidth=0,
                           size=2,
                           jitter=0.3,
                           ax=ax)
    g_real = sns.stripplot(data=metrics_at_threshold_real,
                           x=threshold_type,
                           y=metric_name,
                           hue='group',
                           hue_order=groups_order,
                           palette=dict(zip(groups_order, groups_colors)),
                           dodge=dodge,
                           linewidth=0,
                           size=5,
                           jitter=0.3,
                           marker='X',
                           ax=ax)
    return g_rand, g_real

def make_network_metrics_primary_threshold_figure(
    metrics_df,
    metrics_to_plot=['transitivity', 'assortativity', 'small_worldness'],
    primary_threshold=0.05,
    fontsize=8, figsize=(178*mm, 60*mm), filename='net_metrics_1thresh_4grp_fig.pdf'
):
    metrics_out = {}

    node_label_size = (fontsize*3/4)
    ticksize = (fontsize*3/4)

    gray_color = '0.3'

    matplotlib.rcParams['font.family'] = ['Arial', 'sans-serif']
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = ticksize
    matplotlib.rcParams['ytick.labelsize'] = ticksize

    fig = plt.figure(figsize=figsize, tight_layout=True)
    gs = fig.add_gridspec(1,3, wspace = 0.1, hspace = 0.1)

    for ii, metric in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[ii])
        g = plot_one_threshold_metric_strip_plot_mpl(ax=ax,
                                                     metrics_df=metrics_df,
                                                     metric_name=metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend([], [], frameon=False)
        ax.set_xlabel(None)
        metrics_out[metric] = g

    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return fig, metrics_out

def make_network_metrics_all_thresholds_figure(
    metrics_df,
    metrics_to_plot=['transitivity', 'assortativity', 'small_worldness'],
    fontsize=8, figsize=(178*mm, 60*mm), filename='net_metrics_all_thresh_4grp_fig.pdf'
):
    metrics_out = {}

    node_label_size = (fontsize*3/4)
    ticksize = (fontsize*3/4)

    gray_color = '0.3'

    matplotlib.rcParams['font.family'] = ['Arial', 'sans-serif']
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = ticksize
    matplotlib.rcParams['ytick.labelsize'] = ticksize

    fig = plt.figure(figsize=figsize, tight_layout=True)
    gs = fig.add_gridspec(1,3, wspace = 0.1, hspace = 0.1)

    for ii, metric in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[ii])
        g = plot_multiple_thresholds_metric_line_plot(ax=ax,
                                                       metrics_df=metrics_df,
                                                       metric_name=metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend([], [], frameon=False)
        if metric=='small_worldness':
            ax.set(yscale="log")
        metrics_out[metric] = g

    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return fig, metrics_out


def make_network_metrics_all_thresholds_multiple_subplots_figure(
    metrics_df,
    metrics_to_plot=['transitivity', 'assortativity', 'small_worldness'],
    groups_order=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
    fontsize=8, figsize=(178*mm, 160*mm), filename='net_metrics_all_thresh_multiple_subplots_4grp_fig.pdf'
):
    metrics_out = {}

    node_label_size = (fontsize*3/4)
    ticksize = (fontsize*3/4)

    gray_color = '0.3'

    matplotlib.rcParams['font.family'] = ['Arial', 'sans-serif']
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = ticksize
    matplotlib.rcParams['ytick.labelsize'] = ticksize

    fig = plt.figure(figsize=figsize, tight_layout=True)
    gs = fig.add_gridspec(len(groups_order),3, wspace = 0.1, hspace = 0.1)

    for jj, group in enumerate(groups_order):
        for ii, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[jj, ii])
            g = plot_multiple_thresholds_metric_line_plot(
                ax=ax,
                metrics_df=metrics_df[metrics_df['group']==group],
                metric_name=metric)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend([], [], frameon=False)
            if metric=='transitivity':
                ax.set_ylim([-0.05, 1.0])
            elif metric=='assortativity':
                ax.set_ylim([-1.0, 1.0])
            if metric=='small_worldness':
                ax.set(yscale="log")
                ax.set_ylim([1e-1, 50.0])
            metrics_out[metric] = g

    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return fig, metrics_out

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
