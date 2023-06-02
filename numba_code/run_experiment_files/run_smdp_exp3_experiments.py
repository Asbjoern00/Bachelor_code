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

T = 10**5

#For experiments.
# Test on SMDP-EXP3
env = rs.riverswim(nS = 10, T_max = 10)
_,_,_,gain = utils.VI(env)
smdp_exp3 = ucrlS.SMDP_EXP3(nS = 10, nA = 2, N = 10, P = env.P_smdp)


rew, tau = utils.run_experiment(env, smdp_exp3,T=T)

regret = utils.calc_regret(rew,tau,gain)
smdp_exp3.current_sample_prop
# These are close, which should be a good sign (page 680). 
smdp_exp3.mu_st
smdp_exp3.mu

smdp_exp3.tauhat
smdp_exp3.rhat
smdp_exp3.zero_one

np.sum(smdp_exp3.mu)
np.sum(smdp_exp3.mu_st)



plt.plot(regret)
plt.savefig('C:/Users/andre/OneDrive/KU, MAT-OEK/Bachelor matematik-oekonomi/3. aar/Bachelorprojekt/Out/test')
plt.show()
