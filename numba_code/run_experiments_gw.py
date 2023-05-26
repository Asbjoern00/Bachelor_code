import sys
sys.path.append('C:/Users/andre/Bachelor_code/numba_code')

import numpy as np
import grid_world_class as gw
import UCRL2_L as ucrl
import UCRL_SMDP as ucrlS
import experiment_utils as utils
import importlib
importlib.reload(gw)
importlib.reload(ucrl)
importlib.reload(ucrlS)
importlib.reload(utils)
import matplotlib.pyplot as plt
import gc
import pandas as pd


# Recall for grid world, it only makes sense to take T_max < d
def scenario_generator(nS):
    nS_grid = np.ones(nS)*nS
    T_grid = [x for x in range(2, 3, 4)]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))

n_reps = 4
T = 10**7


# BUS
S = 10*10
nS_list, T_max_list = scenario_generator(nS=S)
nS_list
T_max_list

envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = ucrlS.bus3_wrapper(nS = S, nA= 4, delta=0.05,imprv=1,H=10000,T_max_grid= np.array([1,3,5,8],np.int64))
algos.algorithms[0].reset(0)
algos.play(5,1,2)

utils.run_experiment(envs[-1],algos,T=T)

for algo,env in zip(algos,envs):
    print(algo.T_max_grid,env.T_max)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_bus", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()

# UCRLSMDP

S = 10*10
nS_list, T_max_list = scenario_generator(nS=S)
nS_list
T_max_list


envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 4)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 4)

for algo,env in zip(algos,envs):
    print(algo.T_max,env.T_max)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()


S = 15*15
nS_list, T_max_list = scenario_generator(nS=S)
nS_list
T_max_list



envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 4)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 4)

for algo,env in zip(algos,envs):
    print(algo.T_max,env.T_max)
    
print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()


S = 20*20
nS_list, T_max_list = scenario_generator(nS=S)
nS_list
T_max_list


envs = utils.create_multiple_envs(nS_list, T_max_list, gw.grid_world, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 4, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 4)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 4)

for algo,env in zip(algos,envs):
    print(algo.T_max,env.T_max)
    
print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "gw_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()