import numpy as np
import os
from joblib import delayed, Parallel
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import gc
from numba import njit,jit
import matplotlib

matplotlib.use('Agg')

# An implementation of the Value Iteration algorithm for a given environment 'env' in an average reward setting.
# An arbitrary 'max_iter' is a maximum number of iteration, usefull to catch any error in your code!
# Return the number of iterations, the final value, the optimal policy and the gain.
@jit
def VI(env, max_iter = 10**5, epsilon = 10**(-6)):

	# The variable containing the optimal policy estimate at the current iteration.
	policy = np.zeros(env.nS, dtype=np.int64)
	niter = 0

	# Initialise the value and epsilon as proposed in the course.
	V0 = np.zeros(env.nS,np.float64)
	V1 = np.zeros(env.nS,np.float64)

	# The main loop of the Value Iteration algorithm.
	while True:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R_eq[s, a] + sum([V * p for (V, p) in zip(V0, env.P_eq[s, a])]) # Note Peq instead of P
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy[s] = a
		
		# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
		gain = 0.5*(max(V1 - V0) + min(V1 - V0))
		diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
		if (max(diff) - min(diff)) < epsilon:
			return niter, V0, policy, gain
		else:
			V0 = V1
			V1 = np.zeros(env.nS)
		if niter > max_iter:
			print("No convergence in VI after: ", max_iter, " steps!")
			return niter, V0, policy, gain


@njit(fastmath = True, parallel = True)
def run_experiment(environment, algorithm, T):
    """Function to execute algorithm on environment for T timesteps in the natural process

    Parameters
    ----------
    environment : instance of gridworld
        The instance of gridworld (or potentially other environments) to run algorithm on
    algorithm : Instance of UCRL-variant
        The algorithm to run
    T : int
        Time horizon in the natural process
    write_to_csv : bool
        Whether to write to csv or not. If true will write to npy file in /results_experiment/S_{cardinality of S} formatedded as 'algorithm_T_max_{T_max}.npy'

    Returns
    -------
    reward: np.array, tau: np.array
        Returns np.array of rewards and holding times of length T
    """

    #initialize
    reward = np.zeros(T, dtype=np.float64)
    tau = np.zeros(T, dtype=np.int64)
    # Reset environment and algo 
    environment.reset()
    s = environment.s
    algorithm.reset(s)
    new_s = environment.s

    #init timesteps
    t = 0
    t_prev = np.int64(t)

    while t < T:
        action, _  = algorithm.play(new_s, reward[t_prev], tau[t_prev])
        new_s, reward[t] , tau[t]  = environment.step(action)
        t_prev = t
        t += tau[t]

    return reward,tau

@njit(fastmath = True)
def calc_regret(reward, tau, optimal_gain):
    T_n = np.cumsum(tau)
    regret = T_n*optimal_gain - np.cumsum(reward)
    return regret

@njit(fastmath = True, parallel = True)
def run_algo_n_times(environment, algorithm, T, n_reps):
    _,_,_,gstar = VI(environment)
    ls = []
    for i in np.arange(n_reps):
        rew, tau = run_experiment(environment, algorithm, T)
        reg = calc_regret(rew, tau, gstar)
        ls.append(reg)
    #print(algorithm.__class__.__name__, 'with confidence sets', algorithm.imprv, 'done running on', environment.__class__.__name__, "with T_max =", environment.T_max)
    return ls

def calc_reg_from_ls(ls):
     return np.mean(np.asarray(ls),axis = 0)

def create_multiple_envs(nS_list,T_max_list ,base_env,reps, include_extra_mdp_env=True, **kwargs):
    """Functionality to create list of multiple instances of same base environment

    Parameters
    ----------
    nS_list : lst
        List of N-states for the environments
    T_max_list : lst
        List of T_max for the environments
    base_env : Environment class
        The base environment class to derive from

    Returns
    -------
    lst
        list of environments
    """
    env_list = [base_env(nS = nS, T_max = T_max, **kwargs) for T_max,nS in zip(T_max_list, nS_list)]
    for i in range(reps-1):
          env_list += env_list
    if include_extra_mdp_env:
          env_list += [base_env(nS_list[0], T_max = 1)]
    return env_list

def create_multiple_algos(base_algo, nS_list, T_max_list, **kwargs):
      """Functionality to create list of multiple instances of same base environment

    Parameters
    ----------
    nS_list : lst
        List of N-states for the environments
    T_max_list : lst
        List of T_max for the environments
    base_env : algorithm class
        The base algorithm class to derive from

    Returns
    -------
    lst
        list of algorithm
    """
      return [base_algo(nS = nS, T_max = T_max, **kwargs) for T_max,nS in zip(T_max_list, nS_list)]

def run_multiple_experiments_n_reps(algorithm_list, environment_list, n_reps, T):
    """Runs experiment on lists of algorithms and environments

    Parameters
    ----------
    algorithm_list : lst
        algorithm list
    environment_list : lst
        environment lst
    T : int
        Time to run for
    write_to_csv : bool, optional
        Whether to write to npy file, see the run_experiment function, by default False

    Returns
    -------
    lst
        List of length equal to input lists. Therefore output is indexed as output[i] = algorithm_list[i] run on environment_list[i]
    """
    res_dict = {}
    for algorithm, environment in zip(algorithm_list, environment_list):
        res = run_algo_n_times(environment, algorithm, T, n_reps)
        reg = calc_reg_from_ls(res)
        algo_name = algorithm.__class__.__name__
        if hasattr(algorithm, "imprv"):
                mapper = {0: "", 1:"-L", 2:"-EB"} # 0 = no improvment, 1 = our improvement, 2 = Emma Brunskild
                algo_name += mapper[algorithm.imprv]
        if hasattr(algorithm, "T_max_grid"):
            name = f"{algo_name}, T_max_grid = {[int(T_max) for T_max in algorithm.T_max_grid]}"
        else:
            name = f"{algo_name}, T_max = {int(algorithm.T_max)}"
        res_dict[name] = reg
        print(algo_name)

    return res_dict


def save_dict_as_pd(dict, subdir, experiment_name):
    df = pd.DataFrame.from_dict(dict)
    dir = f"experiment_results/{subdir}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_pickle(f"{dir}{experiment_name}")



def run_experiments_and_save(algorithm_list, environment_list, T, n_reps, subdir, experiment_name):
     regrets = run_multiple_experiments_n_reps(algorithm_list, environment_list, n_reps, T)
     save_dict_as_pd(regrets, subdir, experiment_name)


def plot_mean_regret_from_pd(df, nS, save_to):
    for col in df.columns:
        plt.plot(df[col].values, label = col)
        df.drop(col, 1,inplace = True)
    plt.xlabel("T")
    plt.ylabel("Cummulative regret")
    plt.title(f"{nS} states")
    plt.legend(loc='lower right', prop={'size': 7})
    plt.grid()
    plt.savefig(save_to)
    plt.clf()
    plt.close()
               

# Policy iteration:
def PI(env,max_iter = 10**3):

	# Initialisation of the variables - i.e. step 1 in puterman. 
    policy0 = np.random.randint(env.nA, size = env.nS)
    policy1 = np.zeros(env.nS, dtype = int)

    niter = 0

	# The main loop of the PI algorithm.
    while True:
        niter += 1

		# Policy evaluation step - step 2 - we choose s0=0.
        P_pi = np.array([[env.P_eq[s, policy0[s], ss] for ss in range(env.nS)] for s in range(env.nS)])
        R_pi = np.array([env.R_eq[s, policy0[s]] for s in range(env.nS)])
        Q_s0 = np.eye(env.nS) -  P_pi 
        Q_s0[:,0] = np.ones(env.nS) # Eval P - row s0 equal to 1's
        w = np.linalg.inv(Q_s0) @ R_pi # By following (8.6.8)
        gain = w[0] # according to page 379 in puterman
        V0 = w 
        V0[0] = 0 # note value plays role as bias.
        V1 = np.zeros(env.nS)

		# Updating the policy/ policy improvement.
        for s in range(env.nS):
            for a in range(env.nA):
                temp = env.R_eq[s, a] + sum([V * p for (V, p) in zip(V0, env.P_eq[s, a])])
                if (a == 0) or (temp > V1[s]):
                    V1[s] = temp
                    policy1[s] = a # (This is policy 8.6.2 in puterman)
        # Step 4 in section 8.6 in puterman. 
		# Testing if the policy changed or not.
        test = True
        for s in range(env.nS):
            if policy0[s] != policy1[s] :
                test = False
                break
        if test:
            return niter,V0, policy1, gain # Note V0 corresponds to bias.
        else:
            policy0 = policy1
            policy1 = np.zeros(env.nS, dtype=int)
        if niter > max_iter:
            print("No convergence in PI after: ", max_iter, " steps!")
            return niter,V0, policy1, gain


# Policy iteration SMDP
def PI_SMDP(env,max_iter = 10**3):

	# Initialisation of the variables - i.e. step 1 in puterman. 
    policy0 = np.random.randint(env.nA, size = env.nS)
    policy1 = np.zeros(env.nS, dtype = int)
    niter = 0
	# The main loop of the PI algorithm.
    while True:
        niter += 1

		# Policy evaluation step - step 2 - we choose s0=0.
        P_pi = np.array([[env.P_smdp[s, policy0[s], ss] for ss in range(env.nS)] for s in range(env.nS)])
        R_pi = np.array([env.R_smdp[s, policy0[s]] for s in range(env.nS)])
        tau_pi = np.array([env.tau_bar[s, policy0[s]] for s in range(env.nS)])
        Q_s0 = np.eye(env.nS) -  P_pi 
        Q_s0[:,0] = tau_pi #tau_pi # Eval P - row s0 equal to 1's in MDP, but Tau_bar in SMDP.
        w = np.linalg.lstsq(Q_s0, R_pi,rcond = 10**(-120))[0]   # eq. (8.6.8) in puterman - for dealing with close to zero values.
        gain = w[0] # deduced by page 574 in puterman (referenced equation (8.6.1)). Use s_0=0
        V0 = w
        V0[0] = 0 # note value plays role as bias - but for iterations gain needs to be there.


        V1 = np.zeros(env.nS)


		# Updating the policy/ policy improvement.
        for s in range(env.nS):
            for a in range(env.nA):
                temp = env.R_smdp[s, a] - gain*env.tau_bar[s,a] + sum([V * p for (V, p) in zip(V0, env.P_smdp[s, a])])
                if (a == 0) or (temp > V1[s]):
                    V1[s] = temp
                    policy1[s] = a # (This is policy 8.6.2 in puterman)
        # Step 4 in section 8.6 in puterman. 
		# Testing if the policy changed or not.
        test = True
        for s in range(env.nS):
            if policy0[s] != policy1[s] :
                test = False
                break
        if test:
            return niter,V0, policy1, gain # Note V0 corresponds to bias.
        else:
            policy0 = policy1
            policy1 = np.zeros(env.nS, dtype=int)
        if niter > max_iter:
            print("No convergence in PI after: ", max_iter, " steps!")
            return niter,V0, policy1, gain



# Section including Q-learning (not used)

def alpha(t):
    return 2/((t)**(2/3)+1)


def QL_SMDP(env,T):
    policy = np.zeros(env.nS)
    niter = 1
    epsilon = 1/np.sqrt(niter) # exploration term
    Nk = np.zeros((env.nS, env.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
    Nsas = np.zeros((env.nS, env.nA, env.nS), dtype=int) # Number of occureces of (s, a, s').
	# Initialise the value and epsilon as proposed in the course.
    Q0 = np.zeros((env.nS,env.nA))
    s = env.s # initial state.
    a = np.argmax(Q0[env.s,:])
    for i in range(T):
            print(niter)
            niter +=1 # increment t.
            new_s, reward, tau = env.step(a) # take new action
            Nk[s,a] += 1
            Nsas[s,a,new_s] += 1
            alpha = 2/((Nk[s,a])**(2/3)+1)
            delta = reward + np.max(Q0[new_s,:])-Q0[s,a]
            Q0[s,a] = Q0[s,a] + alpha*delta
            s = new_s # update
            dum = np.random.choice(2,replace=True,p = [epsilon,1-epsilon])
            if dum ==1: # i.e. greedily chosen:
                a = np.argmax(Q0[env.s,:]) # find next action
            else:
                a = np.random.choice(env.nA,replace = True)
            policy = np.argmax(Q0,axis = 1)
    return policy,Q0