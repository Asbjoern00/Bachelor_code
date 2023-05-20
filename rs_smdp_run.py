import numpy as np
import riverswim_class_smdp as rs 
import UCRL2_L as ucrl
import UCRL_SMDP as ucrlS
import experiment_utils as utils
import importlib
importlib.reload(rs)
importlib.reload(ucrl)
importlib.reload(utils)
import matplotlib.pyplot as plt
import gc
import pandas as pd

S = 8
T = 3*10**6

T_max_grid = [2,3,4,5,6,7,8]
n_S = [S for i in range(len(T_max_grid))]
p = 0.90 # for binomial

binomial_envs = utils.create_multiple_envs(nS_list=n_S, T_max_list=T_max_grid, base_env=rs.riverswim, reps = 2, include_extra_mdp_env=False, param = p, distribution = "binomial")
unif_envs = utils.create_multiple_envs(nS_list=n_S, T_max_list=T_max_grid, base_env=rs.riverswim, reps = 2, include_extra_mdp_env=False, distribution = "uniform")

algos_ucrl = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list  = n_S, nA = 2, T_max_list = T_max_grid, imprv = 0)
algos_ucrl_l = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list  = n_S, nA = 2, T_max_list = T_max_grid, imprv = 1)
algos = algos_ucrl + algos_ucrl_l

utils.run_experiments_and_save(algos, binomial_envs, T = T, n_reps=5, subdir="rs_smdp", experiment_name=f"nS_{S}_bin.pkl")
utils.run_experiments_and_save(algos, unif_envs, T = T, n_reps=5, subdir="rs_smdp", experiment_name=f"nS_{S}_unif.pkl")