import numpy as np
import riverswim_class as rs 
import UCRL2_L as ucrl
import UCRL_SMDP as ucrlS
import experiment_utils as utils
import importlib
importlib.reload(rs)
importlib.reload(ucrl)
importlib.reload(utils)
import matplotlib.pyplot as plt

def scenario_generator(nS, grid_size = 4):
    nS_grid = np.ones(grid_size)*nS
    T_grid = [1,round(0.33*nS,0),round(0.66*nS,0),nS]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))

n_reps = 8
T = 10**6

S = 5
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim)
envs += envs
envs += [envs[0]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [T_max_list[0]] , nA = 2)

run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps, save=True)

print("5 done")


S = 10
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim)
envs += envs
envs += [envs[0]]
algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [T_max_list[0]] , nA = 2)

run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps, save=True)

print("10 done")



S = 15
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim)
envs += envs
envs += [envs[0]]
algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [T_max_list[0]] , nA = 2)

run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps, save=True)

print("15 done")
