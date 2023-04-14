import numpy as np

def run_experiment(enviroment, algorithm, T):
    """Function to execute algorithm on environment for T timesteps in the natural process

    Parameters
    ----------
    enviroment : instance of gridworld
        The instance of gridworld (or potentially other environments) to run algorithm on
    algorithm : Instance of UCRL-variant
        The algorithm to run
    T : int
        Time horizon in the natural process

    Returns
    -------
    reward: np.array, tau: np.array
        Returns np.array of rewards and holding times of length T
    """

    #initialize
    reward = np.zeros(T)
    tau = np.zeros(T)

    # Reset environment and algo 
    enviroment.reset()
    algorithm.reset(enviroment.s)
    new_s = enviroment.s

    #init timesteps
    t = int(0) 
    t_prev = int(t)

    while t < T:
        action, _  = algorithm.play(new_s, reward[t_prev], tau[t_prev])
        new_s, reward[t] , tau[t]  = enviroment.step(action)
        t_prev = int(t)
        t += tau[t]
        t = int(t)
    reward = reward[:t]
    tau = tau[:t]

    return reward,tau

def calc_regret(reward, tau, optimal_gain):
    T_n = np.cumsum(tau)
    regret = T_n*optimal_gain - np.cumsum(reward)
    return regret

# An implementation of the Value Iteration algorithm for a given environment 'env' in an average reward setting.
# An arbitrary 'max_iter' is a maximum number of iteration, usefull to catch any error in your code!
# Return the number of iterations, the final value, the optimal policy and the gain.
def VI(env, max_iter = 10**3, epsilon = 10**(-2)):

	# The variable containing the optimal policy estimate at the current iteration.
	policy = np.zeros(env.nS, dtype=int)
	niter = 0

	# Initialise the value and epsilon as proposed in the course.
	V0 = np.zeros(env.nS)
	V1 = np.zeros(env.nS)

	# The main loop of the Value Iteration algorithm.
	while True:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + sum([V * p for (V, p) in zip(V0, env.P_eq[s, a])]) # Note Peq instead of P
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

