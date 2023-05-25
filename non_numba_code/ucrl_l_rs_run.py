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
    T_grid = [x for x in range(2, nS+1, round(nS/5))]
    nS_grid = nS_grid.tolist()
    return list(map(int, nS_grid)), list(map(int,T_grid))

n_reps = 5
T = 5*10**6

S = 5
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 0, include_extra_mdp_env=False)
_,_,_,gstar = utils.VI(envs[0],epsilon=10**(-5)) # Same for all envs 

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")

gc.collect()

gc.collect()

S = 10
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=False)

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")

gc.collect()



S = 15
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=False)

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")


gc.collect()

S =20
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=False)

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")

gc.collect()


gc.collect()


S =25
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=False)

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")

gc.collect()


gc.collect()



S = 30
nS_list, T_max_list = scenario_generator(nS=S)
nS_list,T_max_list = [S], [1]
envs = utils.create_multiple_envs(nS_list, T_max_list, rs.riverswim, 2, include_extra_mdp_env=False)

algos = utils.create_multiple_algos(ucrl.UCRL2_L, [nS_list[0]], [1] , nA = 2)

print(f"running {S}")
utils.run_experiments_and_save(algos, envs, T, gstar, n_reps, "rs_ucrl_l", f"nS_{S}_ucrl_l.pkl")
print(f"done running {S}")

gc.collect()
