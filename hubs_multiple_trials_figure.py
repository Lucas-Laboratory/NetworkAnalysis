import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches

import hubs

mm = 1/25.4 # millimeters in inches

def plot_bar_hub_counts(ax, hubs_all_thresholds_df, bar_colors=['r', 'k']):
    g = sns.barplot(
        data=hubs_all_thresholds_df, 
        x='Region', 
        y='count',
        color=bar_colors[0],
        ax=ax
    )
    ax.legend_ = None
    plt.xlabel(None)
    plt.ylabel('No. trials')
    plt.xticks(rotation=90)
    plt.ylim([0, 10])
    # plt.axvline(x=hubs_all_thresholds_df.shape[0]//3*2-.5, ls=':', c='0.2', lw=0.8)
    plt.box(on=False)
    return g

def make_hub_counts_4groups_figure(thresholds_results_dict,
                                   groups=['naive_male', 'trained_male', 'naive_female', 'trained_female'],
                                   group_colors={'naive_male': 'r', 'naive_female': 'r', 'trained_male': 'r', 'trained_female': 'r'},
                                   fontsize=8,
                                   figsize=(178*mm, 100*mm), filename='hub_counts_4grp_fig.pdf'):

    hubs_all_thresholds = {}
    for group in groups:
        hub_dict_list = []
        for threshold in thresholds_results_dict.keys():
            hub_dict_list.append(thresholds_results_dict[threshold][2][group])
        hubs_all_thresholds[group] = hubs.hub_counts_for_multiple_trials(hub_dict_list)

    groups_out = {}

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
        hub_dict_list = []
        for threshold in thresholds_results_dict.keys():
            hub_dict_list.append(thresholds_results_dict[threshold][2][group])
        hubs_all_thresholds[group] = hubs.hub_counts_for_multiple_trials(hub_dict_list)
        hubs_all_thresholds_sorted_df = hubs_all_thresholds[group].sort_values('count', ascending=False)

        # Bar plot panel
        ax = fig.add_subplot(gs[ii])
        ga = plot_bar_hub_counts(ax=ax,
                                 hubs_all_thresholds_df=hubs_all_thresholds_sorted_df,
                                 bar_colors=[gray_color])
        ga.set_title(group)

        groups_out[group] = ga, hubs_all_thresholds[group]

    gs.tight_layout(fig)
    fig.subplots_adjust(left=0.02, right=1.0, top=1.0, bottom=0.02)
    fig.savefig(filename, bbox_inches='tight')

    return fig, groups_out
