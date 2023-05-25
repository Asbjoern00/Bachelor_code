import numpy as np
import grid_world_class as gw
import UCRL2_L as ucrl
import UCRL_SMDP as ucrlS
import experiment_utils as utils
import importlib
importlib.reload(gw)
importlib.reload(ucrl)
importlib.reload(utils)
import matplotlib.pyplot as plt
import gc
import pandas as pd



n_reps = 5
T = 6*10**6

S = 12**2
nS_list, T_max_list = [S,S,S,S,S,S], [2,3,4,5,6,7]



envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 4)

utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_options", f"gw_nS_{S}.pkl")

S = 16**2
nS_list, T_max_list = [S,S,S,S,S,S], [2,3,4,5,6,7]



envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 4)
utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_options", f"gw_nS_{S}.pkl")
