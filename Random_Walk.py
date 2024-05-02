#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25

@author: Pedro Pinheiro & Khushal Chekuri

Citation: This Python script was based on the Python Script developed for Lab 4 and uses the shapefile
for ND generated by the ND_MAUP jupyter notebook on the ND folder.

All data retrieved 03/25/24:
    https://redistrictingdatahub.org/dataset/north-dakota-block-pl-94171-2020-by-table/
    https://redistrictingdatahub.org/dataset/vest-2020-north-dakota-precinct-boundaries-and-election-results/
    https://redistrictingdatahub.org/dataset/2021-north-dakota-state-senate-approved-plan/

Data available at Google Drive:
https://drive.google.com/drive/folders/1ij4kOO3iKgNRNZtMJNzjVf2X2NHvRIb7?usp=sharing
"""


import geopandas as gpd
import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import Tally
from functools import partial
import time
import random
import pandas as pd


"""
This method adds all the data from a partition to a dictionary.
"""
def add_to_results(results_dict, part):
    for updater in part.updaters.keys():

        # For Tally updaters in the partition, we get the values for each district,
        # and add them individually to the dictionary.
        if isinstance(part.updaters[updater], Tally):
            for dist in part[updater].keys():
                col_name = updater + "_" + str(dist)
                if col_name not in results_dict:
                    results_dict[col_name] = []
                results_dict[col_name].append(part[updater][dist])

        # For Election updaters in the partition, we get the votes each party and for each district,
        # and add them individually to the dictionary.
        elif isinstance(part.updaters[updater], Election):
            for party in ['Democratic', 'Republican']:
                for dist in part[updater].totals_for_party[party].keys():
                    col_name = updater + "_" + party + "_" + str(dist)
                    if col_name not in results_dict:
                        results_dict[col_name] = []
                    results_dict[col_name].append(part[updater].totals_for_party[party][dist])

            # Save the mean_median metric for each election on the dictionary
            metric_name = updater + "_" + "mean_median"
            if metric_name not in results_dict:
                results_dict[metric_name] = []
            results_dict[metric_name].append(part[updater].mean_median())

            # Save the efficiency_gap metric for each election on the dictionary
            metric_name = updater + "_" + "efficiency_gap"
            if metric_name not in results_dict:
                results_dict[metric_name] = []
            results_dict[metric_name].append(part[updater].efficiency_gap())

            # Save the partisan_bias metric for each election on the dictionary
            metric_name = updater + "_" + "partisan_bias"
            if metric_name not in results_dict:
                results_dict[metric_name] = []
            results_dict[metric_name].append(part[updater].partisan_bias())
    
        elif updater == "cut_edges":
            if updater not in results_dict:
                results_dict[updater] = []
            results_dict[updater].append(len(part[updater]))

# Specify seed:
random.seed(123456)

# Read Geodataframe from Shapefile.
gdf = gpd.read_file("./data_cleaning/ND/ND.shp")

# Plot to check if shape is expected.
gdf.plot()
plt.show()

# Get Graph from geodataframe
nd_graph = Graph.from_geodataframe(gdf)

# Initial updaters to contain the cut_edges, district population per division.
my_updaters = {
    "cut_edges": updaters.cut_edges,
    "population": Tally("TOTPOP", alias="population"),
    "white population": Tally("WHITE", alias = "white population"),
    "black population": Tally("BLACK", alias = "black population"),
    "native population": Tally("AMIN", alias = "native population"),
    "asian population": Tally("ASIAN", alias = "asian population"),
    "nhpi population": Tally("NHPI", alias = "nhpi population"),
    "other population": Tally("OTHER", alias = "other population"),
    "2 or more population": Tally("2MORE", alias = "2 or more population"),

    "latino population": Tally("HISP", alias = "latino population"),
    "non hisp white population": Tally("NH_WHITE", alias = "non hisp white population"),
    "non hisp black population": Tally("NH_BLACK", alias = "non hisp black population"),
    "non hisp native population": Tally("NH_AMIN", alias = "non hisp native population"),
    "non hisp asian population": Tally("NH_ASIAN", alias = "non hisp asian population"),
    "non hisp nhpi population": Tally("NH_NHPI", alias = "non hisp nhpi population"),
    "non hisp other population": Tally("NH_OTHER", alias = "non hisp other population"),
    "non hisp 2 or more population": Tally("NH_2MORE", alias = "non hisp 2 or more population"),

    "voting age population": Tally("VAP", alias="voting age population"),
    "white voting age population": Tally("WVAP", alias = "white voting age population"),
    "black voting age population": Tally("BVAP", alias = "black voting age population"),
    "native voting age population": Tally("AMINVAP", alias = "native voting age population"),
    "asian voting age population": Tally("ASIANVAP", alias = "asian voting age population"),
    "nhpi voting age population": Tally("NHPIVAP", alias = "nhpi voting age population"),
    "other voting age population": Tally("OTHERVAP", alias = "other voting age population"),
    "2 or more voting age population": Tally("2MOREVAP", alias = "2 or more voting age population"),

    "latino voting age population": Tally("HVAP", alias = "latino voting age population"),
    "non hisp white voting age population": Tally("NH_WVAP", alias = "non hisp white voting age population"),
    "non hisp black voting age population": Tally("NH_BVAP", alias = "non hisp black voting age population"),
    "non hisp native voting age population": Tally("NH_AMINVAP", alias = "non hisp native voting age population"),
    "non hisp asian voting age population": Tally("NH_ASIAVAP", alias = "non hisp asian voting age population"),
    "non hisp nhpi voting age population": Tally("NH_NHPIVAP", alias = "non hisp nhpi voting age population"),
    "non hisp other voting age population": Tally("NH_OTHEVAP", alias = "non hisp other voting age population"),
    "non hisp 2 or more voting age population": Tally("NH_2MORVAP", alias = "non hisp 2 or more voting age population"),

    "hisp white population": Tally("H_WHITE", alias = "hisp white population"),
    "hisp black population": Tally("H_BLACK", alias = "hisp black population"),
    "hisp native population": Tally("H_AMIN", alias = "hisp native population"),
    "hisp asian population": Tally("H_ASIAN", alias = "hisp asian population"),
    "hisp nhpi population": Tally("H_NHPI", alias = "hisp nhpi population"),
    "hisp other population": Tally("H_OTHER", alias = "hisp other population"),
    "hisp 2 or more population": Tally("H_2MORE", alias = "hisp 2 or more population"),

    "hisp white voting age population": Tally("H_WVAP", alias = "hisp white voting age population"),
    "hisp black voting age population": Tally("H_BLACK", alias = "hisp black voting age population"),
    "hisp native voting age population": Tally("H_AMINVAP", alias = "hisp native voting age population"),
    "hisp asian voting age population": Tally("H_ASIANVAP", alias = "hisp asian voting age population"),
    "hisp nhpi voting age population": Tally("H_NHPIVAP", alias = "hisp nhpi voting age population"),
    "hisp other voting age population": Tally("H_OTHERVAP", alias = "hisp other voting age population"),
    "hisp 2 or more voting age population": Tally("H_2MOREVAP", alias = "hisp 2 or more voting age population")
}

# Build elections objects.
elections = [
    Election("PRE20", {"Democratic": "PRES20D", "Republican": "PRES20R"}),
    Election("HAL20", {"Democratic": "HAL20D", "Republican": "HAL20R"}),
    Election("GOV20", {"Democratic": "GOV20D", "Republican": "GOV20R"}),
    Election("AUD20", {"Democratic": "AUD20D", "Republican": "AUD20R"}),
    Election("TRE20", {"Democratic": "TRE20D", "Republican": "TRE20R"}),
    Election("PSC20", {"Democratic": "PSC20D", "Republican": "PSC20R"})
]

# Add elections to updaters dict
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

# Create initial partion using updaters and the Congressional Districting ("SLDU_2021") as the assignment.
initial_partition = Partition(
    graph=nd_graph,
    assignment="SLDU_2021",
    updaters=my_updaters
)

# Get total population by summing districs population.
tot_pop = sum([initial_partition["population"][dist] for dist in initial_partition["population"].keys()])
print("Total Population: ", tot_pop)

# Get total population by summing districs population.
num_dist = len(initial_partition)
ideal_pop = tot_pop/num_dist
print("Number of Districts: ", num_dist)
print("Ideal District Population: ", ideal_pop)

pop_tolerance = 0.05
# Random walk proposal using recom
rw_proposal = partial(
    proposals.recom, # how you choose a next districting plan
    pop_col = "TOTPOP", # What data describes population? Column from original graph
    pop_target = ideal_pop, # What the target/ideal population is for each district 
    epsilon = pop_tolerance,
    node_repeats = 1 # number of times to repeat bipartition. Can increase if you get a BipartitionWarning
)

# Create population constraint using population percentage
population_constraint = constraints.within_percent_of_ideal_population(
    initial_partition=initial_partition,
    percent=pop_tolerance,
    pop_key="population"
)

# Create a short random walk using proposal and contraint to test.
total_steps = 50000
random_walk = MarkovChain(
    proposal = rw_proposal, 
    constraints = [population_constraint],
    accept = accept.always_accept, # Accept every proposed plan that meets the population constraints
    initial_state = initial_partition, 
    total_steps = total_steps
)

# Create dict to save results and add initial_partition
rw_results_dict = {}
add_to_results(rw_results_dict, initial_partition)

# Get start time
start_time = time.time()
print("Running Random Walk...")

# Run random walk
s = 1
for part in random_walk:
    add_to_results(rw_results_dict, part)
    if (total_steps >= 100 and (s % (total_steps//100) == 0)):
        print(f"Progress = {s / (total_steps//100)}%")
    elif (total_steps >= 1000 and (s % (total_steps//1000) == 0)):
        print(f"Progress = {s / (total_steps//1000) / 10}%")
    elif (total_steps >= 10000 and (s % (total_steps//10000) == 0)):
        print(f"Progress = {s / (total_steps//10000) / 100}%")

    # Save the results to a file at every 10% of the run to prevent accidental data loss.
    if (total_steps >= 10 and (s % (total_steps//10) == 0)):
        rw_data_df = pd.DataFrame(rw_results_dict)
        rw_data_df.to_csv(f'rw_data_{s}_steps.csv', index=False)

    s += 1

# Get end time
end_time = time.time()
print()
print("The time of execution of above program is :",
      (end_time-start_time)/60, "mins")