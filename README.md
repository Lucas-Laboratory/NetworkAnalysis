# Network analysis

## Table of contents

* [Introduction](#introduction)
* [Setup](#setup)
* [File overview](#file-overview)
* [du Plessis et al. (2022)](#du-plessis-et-al-2022)

## Introduction

This repository contains code for aiding with functional brain network analysis. The .py files are collections of functions that are mostly wrappers around the graph theoretic network analysis tools provided by Brain Connectivity Toolbox and Networkx. See the file overview for brief descriptions of the contents of each of these .py files.

The network analysis and figure generation code used in du Plessis et al. (2022) can be found in two Jupyter notebooks, which guide through the recreation of that analysis. Some of the analysis is time consuming (generating and analyzing 1000 random graphs per group, per correlation threshold), so pre-computed network metrics are provided in .dat files in the du_Plessis_et_al_2022 folder. To analyze new, raw data, the format for the 'data' is a pandas DataFrame, the format information is described in the 'funcs.load_data' functions.

## Setup

The requirements.txt file provides python packages used in all functions in the .py files.

Suggested setup of python environment (python 3.9.6 was used to run code for the paper):

```bash
# create an env
python -m venv duPlessis2022

# use the env
source duPlessis2022/bin/activate

# Install required packages
pip install -r requirements.txt
```

The main examples for using the functions provided here are the Jupyter notebooks that recreate the data analysis and network figures from du Plessis et al. (2022).

Suggested method for starting the jupyter notebooks:

```bash
# use the env created above
source duPlessis2022/bin/activate

# Start Jupyter
jupyter notebook
```

## File overview

### funcs.py

Main analysis, with general functions for:

* Loading data
* Computing correlation and adjacency matrices
* Network-level and node-level metrics provided by Brain Connectivity Toolbox
* Helper functions for generating random networks and computing metrics across multiple correlation thresholds
* Small-worldness calculation

### clusters.py

Clustering related functions for:

* Markov clustering, with modularity analysis for determining inflation parameters
* Rearranging data by clusters

### hubs.py

Helper functions for gathering the centrality measures used to find hubs

Then there are several files with the functions for making the figures in du Plessis et al. (2022):

### figures.py

Plotting network graphs and metric scatter plots

### correlations_figure.py

Plotting correlation matrices

### cluster_graphs_figure.py

Everything for laying out and plotting figure 4, the network graph and centrality measures for all 4 groups

### metrics_figures.py

Functions for plotting the network-level metrics in the supplemental figures

### hubs_multiple_trials_figure.py

Plotting the bar plot of hub counts across multiple thresholds in the supplemental figure

## du Plessis et al. (2022)

These two Jupyter notebooks perform the analysis and generate basic versions of the figures found in du Plessis et al. (2022).

    du_Plessis_et_al_2022_generate_data.ipynb
    du_Plessis_et_al_2022_figures.ipynb

The 'generate_data' notebook can be followed to perform all random network generation and analysis. The cells there that sample 1000 random networks for each group in the data take several hours to run. For convenience the completed random network generation and analysis results are provided in the 'du_Plessis_et_al_2022/' folder, in .dat files, one for each group. The calculation of network measures for multiple thresholds on the real data is found in a cell in both notebooks, and takes ~5 minutes - or it can be loaded from the thresholds_results_dict.dat file.
