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

########################################################################
T = 5*10**6
n_reps = 10
# ACTUAL RUNS

# RUN: 10 STATE RIVERSWIM WITH OPTIONS AND N=5) 

for n in [5,10,15,20]: 
    S = 10
    T_max_grid=[S-5,S,S+5]
    for t in T_max_grid:
        env = rs.riverswim(nS=S, T_max=t)
        smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = n, P = env.P_smdp)
        utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{n}.pkl")

    print('done','nS',S,'N',n)

    # RUN: 20 STATE RIVERSWIM WITH OPTIONS AND N=20)

    # Read other data first. 

    S = 20
    T_max_grid=[S-5,S,S+5]
    for t in T_max_grid:
        env = rs.riverswim(nS=S, T_max=t)
        smdp_exp3 = ucrlS.SMDP_EXP3(nS = S, nA = 2, N = n, P = env.P_smdp)
        utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs", experiment_name=f"smpdexp3_ns_{S}_Tmax_{t}_N{n}.pkl")
    print('done','nS',S,'N',n)



########################################################################
# RUN 2 - use tau not tau hat
########################################################################
T = 5*10**6
n_reps = 10
# ACTUAL RUNS

# RUN: 10 STATE RIVERSWIM WITH OPTIONS AND N=5) 

for n in [5,10,15,20]: 
    S = 10
    T_max_grid=[S-5,S,S+5]
    for t in T_max_grid:
        env = rs.riverswim(nS=S, T_max=t)
        smdp_exp3 = ucrlS.SMDP_EXP3_2(nS = S, nA = 2, N = n, P = env.P_smdp)
        utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs_tau", experiment_name=f"smpdexp3_tau_ns_{S}_Tmax_{t}_N{n}.pkl")

    print('done','nS',S,'N',n)

    # RUN: 20 STATE RIVERSWIM WITH OPTIONS AND N=20)

    # Read other data first. 

    S = 20
    T_max_grid=[S-5,S,S+5]
    for t in T_max_grid:
        env = rs.riverswim(nS=S, T_max=t)
        smdp_exp3 = ucrlS.SMDP_EXP3_2(nS = S, nA = 2, N = n, P = env.P_smdp)
        utils.run_experiments_and_save([smdp_exp3], [env], T = T, n_reps=n_reps, subdir = "smdpexp3_rs_tau", experiment_name=f"smpdexp3_tau_ns_{S}_Tmax_{t}_N{n}.pkl")
    print('done','nS',S,'N',n)