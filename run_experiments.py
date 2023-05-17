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

def scenario_generator(nS):
    nS_grid = np.ones(nS)*nS
    T_grid = [x+1 for x in range(1, nS+5, round(nS/6))]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))

n_reps = 8
T = 5*10**1

S = 5
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
_,_,_,gstar = utils.VI(envs[0],epsilon=10**(-5))

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 2)

for algo, env in zip(algos, envs):
    print(algo.T_max, env.T_max)
print("running 5")
run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps = n_reps)
utils.mean_regret_from_dict(run, g_star=gstar, save=True, sub_dir="rs_ucrl_l", file_name = f"nS_{S}")
print("5 done")

del run
gc.collect()

S = 10
nS_list, T_max_list = scenario_generator(nS=S)
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)

algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 2)

print("running 10")
_,_,_,gstar = utils.VI(envs[0],epsilon=10**(-5))
run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps = n_reps)
utils.mean_regret_from_dict(run, g_star=gstar, save=True, sub_dir="rs_ucrl_l", file_name = f"nS_{S}")
del run
gc.collect()

print("10 done")


print("running 15")
S = 15
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 2)

_,_,_,gstar = utils.VI(envs[0],epsilon=10**(-5))
run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps = n_reps)
utils.mean_regret_from_dict(run, g_star=gstar, save=True, sub_dir="rs_ucrl_l", file_name = f"nS_{S}")
del run
gc.collect()

print("15 done")

print("running 20")
S = 20
nS_list, T_max_list = scenario_generator(nS=S)

envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=True)
algos = utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2)
algos += utils.create_multiple_algos(ucrlS.UCRL_SMDP, nS_list, T_max_list , nA = 2, imprv = True)
algos += utils.create_multiple_algos(ucrl.UCRL2, [nS_list[0]], [1] , nA = 2)

_,_,_,gstar = utils.VI(envs[0],epsilon=10**(-5))
run = utils.run_multiple_experiments_n_reps(algos, envs, T, n_reps = n_reps)
utils.mean_regret_from_dict(run, g_star=gstar, save=True, sub_dir="rs_ucrl_l", file_name = f"nS_{S}")
print("20 done")

del run
gc.collect()