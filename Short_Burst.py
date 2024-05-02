#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Metric Geometry and Gerrymandering Group

Created on Mon Apr 29

@author: Pedro Pinheiro & Khushal Chekuri

Citation: This Python script was based on the Python Script in https://github.com/vrdi/shortbursts-gingles/blob/main/state_experiments/sb_runs.py

Data available at Google Drive:
https://drive.google.com/drive/folders/1ij4kOO3iKgNRNZtMJNzjVf2X2NHvRIb7?usp=sharing
"""


import geopandas as gpd
import numpy as np
import pickle
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally
from gingleator import Gingleator
import multiprocessing

# Instatiate variables
STATE = "ND"
NUM_DISTRICTS = 47
POP_COL = "TOTPOP"
MIN_POP_COL = "AMINVAP"

POP_TOT = 0.05
BURST_LENS = [5, 10, 15] # Burst lenghts to run shortburts.
SCORE_FUNCT = Gingleator.num_opportunity_dists # Score function: number of opportunity districts
THRESHOLDS = [0.4, 0.45, 0.5] # Threshold to be considered an opportunity district
ITERS = 40000 # Total number of steps in the Markov Chain

MAX_PROCESSES = 9 # Max number of paralel processes

# Load shapefile and graph
gdf = gpd.read_file("./data_cleaning/ND/ND.shp")
nd_graph = Graph.from_geodataframe(gdf)

# Get updaters that contain the desired population information:
my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "AMINVAP": Tally("AMINVAP")}

# Initialize initial partition
init_partition = Partition(
    graph=nd_graph,
    assignment="SLDU_2021",
    updaters=my_updaters
)

"""
This method runs a shortburst for a specific burt lenght and score threshold
"""
def process_sb_obs(threshold, burst_len):
    params = f"{STATE}_dists{NUM_DISTRICTS}_{MIN_POP_COL}opt_{POP_TOT:.1%}_{ITERS}_sbl{burst_len}_score{SCORE_FUNCT.__name__}_{threshold}"
    print(f"Started chain for threshold = {threshold} and burst_len= {burst_len}.\n Params: {params}", flush=True)

    # Number of bursts for this run
    num_bursts = ITERS//burst_len
    
    gingles = Gingleator(init_partition, pop_col=POP_COL,
                         threshold=threshold, score_funct=SCORE_FUNCT, epsilon=POP_TOT,
                         minority_perc_col="{}_perc".format(MIN_POP_COL))

    # This initializes a column with the percentage of the VAP of interest in each district
    gingles.init_minority_perc_col(MIN_POP_COL, "VAP", "{}_perc".format(MIN_POP_COL))

    # Runs the short burst
    sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=burst_len,
                                     maximize=True, verbose=True)

    print(f"Finished chain for threshold = {threshold} and burst_len= {burst_len}", flush=True)

    print(f"Saving results for threshold = {threshold} and burst_len= {burst_len}", flush=True)

    # Save the run scores to a numpy file:
    f_out_res = f"data_sb/{params}.npy"
    np.save(f_out_res, sb_obs[1])

    # Save information about the best partition to a pickle file:
    f_out_stats = f"data_sb/{params}.p"
    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "AMINVAP": sb_obs[0][0]["AMINVAP"]}

    with open(f_out_stats, "wb") as f_out:
        pickle.dump(max_stats, f_out)

# This will start the processes in paralel for all the combinations of burst lenghts and threshold
if __name__ == '__main__':
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        pool.starmap(process_sb_obs, [(th, bl) for th in THRESHOLDS for bl in BURST_LENS])