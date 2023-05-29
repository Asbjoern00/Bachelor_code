import experiment_utils as utils
import riverswim_class as rs
import UCRL_SMDP as surcl
import numpy as np
def lambda_grid(T_bound):
    stop = 0
    T_max_grid = []
    while T_bound > 2**stop:
        stop += 1
        T_max_grid.append(2**stop)
    
    T_max_grid = np.array(T_max_grid, dtype=np.int64)
    T_max_grid = np.minimum(T_bound, T_max_grid)
    return T_max_grid


### NS = 50, T_max = 22

nS = 50
T_max = 22
Hs = [10,100,1000]
T_max_grid=lambda_grid(nS*T_max)
n_reps = 20
for H in Hs:
    bus = surcl.bus3_wrapper(nS, nA=2, delta=0.05, H=H, imprv=1, T_max_grid=T_max_grid)
    env = rs.riverswim(nS=nS, T_max=T_max)
    utils.run_experiments_and_save([bus], [env], T = 10**7, n_reps=n_reps, subdir = "bus_opt_rs", experiment_name=f"bus_ns_{nS}_opt_H_{H}")


    agg_hist = utils.mean_hist_probs(env, bus, T = 10**7, n_reps = n_reps)
    hist_probs_dict = {}
    for i in range(len(T_max_grid)):
        hist_probs_dict[f"T_max = {T_max_grid[i]}"] = agg_hist[:,i]
    utils.save_dict_as_pd(hist_probs_dict, subdir = "bus_opt_rs", experiment_name = f"concentration_{nS}_opt_H_{H}")



### NS = 50, T_max = 22

nS = 25
T_max = 14
Hs = [10,100,1000]
T_max_grid=lambda_grid(nS*T_max)
n_reps = 20
for H in Hs:
    bus = surcl.bus3_wrapper(nS, nA=2, delta=0.05, H=H, imprv=1, T_max_grid=T_max_grid)
    env = rs.riverswim(nS=nS, T_max=T_max)
    utils.run_experiments_and_save([bus], [env], T = 10**7, n_reps=n_reps, subdir = "bus_opt_rs", experiment_name=f"bus_ns_{nS}_opt_H_{H}")


    agg_hist = utils.mean_hist_probs(env, bus, T = 10**7, n_reps = n_reps)
    hist_probs_dict = {}
    for i in range(len(T_max_grid)):
        hist_probs_dict[f"T_max = {T_max_grid[i]}"] = agg_hist[:,i]
    utils.save_dict_as_pd(hist_probs_dict, subdir = "bus_opt_rs", experiment_name = f"concentration_{nS}_opt_H_{H}")
