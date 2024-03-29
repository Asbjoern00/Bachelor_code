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
import matplotlib.pyplot as plt
import gc
import pandas as pd

def scenario_generator(nS):
    nS_grid = np.ones(nS)*nS
    T_grid = [x for x in range(2, nS+3, 5)]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))

n_reps = 20
T = 10**7



S = 5
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()


S = 10
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()

S = 15
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)
for env, algo in zip(envs, algos):
    print((env.T_max,algo.T_max))


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()

S = 20
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()



S = 25
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)
for env, algo in zip(envs, algos):
    print((env.T_max,algo.T_max))


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()



S = 50
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)
for env, algo in zip(envs, algos):
    print((env.T_max,algo.T_max))


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()

##########################################
# RIVERSWIM FOR COMPARISONS OF SMDP-EXP3.
def scenario_generator(nS):
    nS_grid = np.ones(nS)*nS
    T_grid = [9,10,11]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))


n_reps = 10
T = 5*10**6


S = 10
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)
for env, algo in zip(envs, algos):
    print((env.T_max,algo.T_max))


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()




### nS = 10, Different T_max. 
def scenario_generator(nS):
    nS_grid = np.ones(nS)*nS
    T_grid = [19,20,21]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))


S = 20
nS_list, T_max_list = scenario_generator(nS=S)



envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
envs += [envs[-1]]

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 1)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = 0)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 0, nA = 2)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , imprv = 1, nA = 2)
for env, algo in zip(envs, algos):
    print((env.T_max,algo.T_max))


print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, n_reps, "rs_ucrl_l", f"nS_{S}_correct_confidence.pkl")
print(f"done running {S}")

gc.collect()