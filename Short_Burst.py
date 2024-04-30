import geopandas as gpd
import numpy as np
import pickle
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally
from gingleator import Gingleator
import multiprocessing

STATE = "ND"
NUM_DISTRICTS = 47
POP_COL = "TOTPOP"
MIN_POP_COL = "AMINVAP"

POP_TOT = 0.05
BURST_LENS = [5, 10, 15]

SCORE_FUNCT = Gingleator.num_opportunity_dists
THRESHOLDS = [0.4, 0.45, 0.5]

ITERS = 40000
MAX_PROCESSES = 9

gdf = gpd.read_file("./data_cleaning/ND/ND.shp")
nd_graph = Graph.from_geodataframe(gdf)

my_updaters = {"population" : Tally(POP_COL, alias="population"),
               "VAP": Tally("VAP"),
               "AMINVAP": Tally("AMINVAP")}

init_partition = Partition(
    graph=nd_graph,
    assignment="SLDU_2021",
    updaters=my_updaters
)

def process_sb_obs(threshold, burst_len):
    params = f"{STATE}_dists{NUM_DISTRICTS}_{MIN_POP_COL}opt_{POP_TOT:.1%}_{ITERS}_sbl{burst_len}_score{SCORE_FUNCT.__name__}_{threshold}"
    print(f"Started chain for threshold = {threshold} and burst_len= {burst_len}.\n Params: {params}", flush=True)

    num_bursts = ITERS//burst_len
    
    gingles = Gingleator(init_partition, pop_col=POP_COL,
                         threshold=threshold, score_funct=SCORE_FUNCT, epsilon=POP_TOT,
                         minority_perc_col="{}_perc".format(MIN_POP_COL))

    gingles.init_minority_perc_col(MIN_POP_COL, "VAP", "{}_perc".format(MIN_POP_COL))

    sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=burst_len,
                                     maximize=True, verbose=True)
    
    print(f"Finished chain for threshold = {threshold} and burst_len= {burst_len}", flush=True)

    print(f"Saving results for threshold = {threshold} and burst_len= {burst_len}", flush=True)

    f_out_res = f"data_sb/{params}.npy"
    np.save(f_out_res, sb_obs[1])

    f_out_stats = f"data_sb/{params}.p"
    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "AMINVAP": sb_obs[0][0]["AMINVAP"]}

    with open(f_out_stats, "wb") as f_out:
        pickle.dump(max_stats, f_out)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        pool.starmap(process_sb_obs, [(th, bl) for th in THRESHOLDS for bl in BURST_LENS])