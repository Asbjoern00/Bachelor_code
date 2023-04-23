import numpy as np
# PURPOSE: IMPLEMENTATION OF EPSILON-GREEDY ALGORITHM. 
class GREEDY():
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        # The "counter" variables:
        self.Nsa = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        
        #Counter variables that might be unnecessary
        self.i = 1
        #For detecting last action
        self.last_action = -1
        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        
        # The current policy (updated at each decision step).
        self.policy = np.zeros((self.nS,), dtype=int)
    
    def eps(self):
        return 1/np.sqrt(self.i) # return 1/sqrt(decision step)
    def VI(self,max_iter = 10**3):
        # The variable containing the optimal policy estimate at the current iteration.
        policy = np.zeros(self.nS, dtype=int)
        niter = 0
        epsilon = 1/np.sqrt(self.i)

        # Initialise the value and epsilon as proposed in the course.
        V0 = np.zeros(self.nS)
        V1 = np.zeros(self.nS)

        # The main loop of the Value Iteration algorithm.
        while True:
            niter += 1
            for s in range(self.nS):
                for a in range(self.nA):
                    temp = self.hatR[s, a] + sum([V * p for (V, p) in zip(V0, self.hatP[s, a])]) # Note Peq instead of P
                    if (a == 0) or (temp > V1[s]):
                        V1[s] = temp
                        policy[s] = a
            
            # Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
            gain = 0.5*(max(V1 - V0) + min(V1 - V0))
            diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
            if (max(diff) - min(diff)) < epsilon:
                return policy
            else:
                V0 = V1
                V1 = np.zeros(self.nS)
            if niter > max_iter:
                print("No convergence in VI after: ", max_iter, " steps!")
                return policy

    def play(self,state,reward,tau=1): # here just time step/decision step.
        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward

		# Update estimates and confidence intervals.
        for s in range(self.nS):
            for a in range(self.nA):
                    self.hatR[s, a] = self.Rsa[s, a] / max((1, self.Nsa[s, a]))
                    for ss in range (self.nS):
                        self.hatP[s, a, ss] = self.Nsas[s, a, ss] / max((1, self.Nsa[s, a]))
		
		# Run VI and get new optimisitc greedy policy.
        policy = self.VI()
        dum = np.random.choice(2,replace=True,p = [self.eps(),1-self.eps()])
        if dum ==1: # i.e. greedily chosen:
            action = policy[state]            
        else:
            action = np.random.choice(env.nA,replace = True)
        tau = 1
		# Update the variables:
        self.Nsa[state, action] += 1
        self.s = state
        self.last_action = action
        self.i += 1 
        return action, self.policy
    
    def reset(self, init=0):
            # The "counter" variables:
            self.Nsa = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
            self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
            self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
            
            # The "estimates" variables:
            self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
            self.hatR = np.zeros((self.nS, self.nA))
            
            # The current policy (updated at each episode).
            self.policy = np.zeros((self.nS,), dtype=int)

            # Set the initial state and last action:
            self.s = init
            self.last_action = -1
            self.i = 1
        



import riverswim_class as rs 
import importlib
importlib.reload(rs)
import experiment_utils as utils
importlib.reload(utils)
import matplotlib.pyplot as plt

env = rs.riverswim(nS=3,T_max = 1)

eps_greed = GREEDY(nS = 3,nA=2)
reward_sucrl,tau_sucrl = utils.run_experiment(env, eps_greed, T = 10**5)
_,_,_,gstar = utils.VI(env)

regret_sucrl = utils.calc_regret(reward=reward_sucrl, tau = tau_sucrl, optimal_gain=gstar)
plt.plot(regret_sucrl, label = "e-greedy")
plt.show()

