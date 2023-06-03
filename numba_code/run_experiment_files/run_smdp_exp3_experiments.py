import sys
sys.path.append('C:/Users/andre/Bachelor_code/numba_code')

import numpy as np
import riverswim_class as rs 
import UCRL2_L as ucrl
import UCRL_SMDP as ucrlS
import experiment_utils as utils
import importlib
importlib.reload(rs)
importlib.reload(ucrl)
importlib.reload(utils)
importlib.reload(ucrlS)

import matplotlib.pyplot as plt
import gc
import pandas as pd

# ACTUAL RUNS

# RUN: 10 STATE RIVERSWIM WITH OPTIONS AND N=4) 
T = 500
N = 4
N

S = 10
T_max_grid=[S-1,S,S+1]
n_reps = 7
for t in T_max_grid:
    env = rs.riverswim(nS=S, T_max=t)
    smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = N, P = env.P_smdp)
    utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{N}.pkl")




# RUN: 20 STATE RIVERSWIM WITH OPTIONS AND N=4)
# Read other data first. 

S = 20
T_max_grid=[S-1,S,S+1]
n_reps = 7
for t in T_max_grid:
    env = rs.riverswim(nS=S, T_max=t)
    smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = 8, P = env.P_smdp)
    utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{N}.pkl")
    smdp_exp3.current_sample_prop


# ACTUAL RUNS

# RUN: 10 STATE RIVERSWIM WITH OPTIONS AND N=10) 
T = 5*10**4
N = 10


S = 10
T_max_grid=[S-1,S,S+1]
n_reps = 7
for t in T_max_grid:
    env = rs.riverswim(nS=S, T_max=t)
    smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = N, P = env.P_smdp)
    utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{N}.pkl")



# RUN: 20 STATE RIVERSWIM WITH OPTIONS AND N=10)
# Read other data first. 

S = 20
T_max_grid=[S-1,S,S+1]
n_reps = 7
for t in T_max_grid:
    env = rs.riverswim(nS=S, T_max=t)
    smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = 8, P = env.P_smdp)
    utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{N}.pkl")



