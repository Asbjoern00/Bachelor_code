import experiment_utils as utils
import riverswim_class_smdp as rs
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


### NS = 8, T_max = 22

nS = 8
T_max_list = [2,4,6]
H = 8*2
n_reps = 20
for T_max in T_max_list:
    T_max_grid=lambda_grid(nS*T_max)
    bus = surcl.bus3_wrapper(nS, nA=2, delta=0.05, H=H, imprv=1, T_max_grid=T_max_grid)
    sucrl_bound = surcl.UCRL_SMDP(nS, 2, T_max=nS*T_max, imprv = 1)
    env = rs.riverswim(nS=nS, T_max=T_max, distribution=1)
    utils.run_experiments_and_save([bus,sucrl_bound], [env,env], T = 5*10**6, n_reps=n_reps, subdir = "bus_smdp_rs", experiment_name=f"bus_ns_{nS}_smdp_T_max_{T_max}")


    agg_hist = utils.mean_hist_probs(env, bus, T = 5*10**6, n_reps = n_reps)
    hist_probs_dict = {}
    for i in range(len(T_max_grid)):
        hist_probs_dict[f"T_max = {T_max_grid[i]}"] = agg_hist[:,i]
    utils.save_dict_as_pd(hist_probs_dict, subdir = "bus_smdp_rs", experiment_name = f"concentration_{nS}_opt_T_max_{T_max}")
