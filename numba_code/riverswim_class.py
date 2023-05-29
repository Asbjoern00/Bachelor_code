# Implementation of riverswim as an SMDP (/MDP with options)
import numpy as np
from numba import int64, float64
from numba.experimental import jitclass
# We let ourselves be inspired by some of the provided source code from the course OReL.
# But we modify it to handle options. 


spec = [
	('nS', int64),
	('T_max', int64),
	('nA', int64),
	('s', int64),
	('map', int64[:,:]),
	('P', float64[:,:,:]),
	('tau', float64[:,:,:]),
	('P_eq', float64[:,:,:]),
    ('P_smdp', float64[:,:,:]),
	('tau_bar', float64[:,:]),
  	('R', float64[:,:]),
	('R_eq', float64[:,:]),
    ('R_smdp', float64[:,:])
]

@jitclass(spec = spec)
class riverswim:
    """Riwerswim class

    Attributes
    ----------
    P : np.array
        Transition probability matrix of the underlying MDP
    P_smdp : np.array
        Transition probability matrix of the smdp induced on the MDP by holding times
    p_eq : np.array
        The equivalent transition probability matrix of the smdp

    """
	
    def __init__(self, nS,T_max):
        self.nS = nS
        self.nA = 2 # two options.
        self.P = np.zeros((nS, 2, nS), np.float64) # Transition probabilities.
        self.P_eq = np.zeros((nS, 2, nS), np.float64) # Transition probabilities in equivalent MDP.
        
        self.T_max = T_max
        
        self.tau = np.full((nS, 2, self.T_max), 1/self.T_max, np.float64) # Holding times. For now assumed uniform
 
        

        # action probs.
        for s in np.arange(nS):
            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s + 1] = 0.4
            elif s == nS - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s - 1] = 0.4
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s] = 0.55
                self.P[s, 1, s + 1] = 0.4
                self.P[s, 1, s - 1] = 0.05
        
        # We build the reward matrix R (same as simple implementation)
        self.R = np.zeros((nS, 2), np.float64)
        self.R[0, 0] = 0.05
        self.R[nS - 1, 1] = 1
        # We (arbitrarily) set the initial state in the leftmost position.
        self.s = 0

        self.P_smdp = np.zeros((nS,2,nS), np.float64) # smdp probabilities.
        self.R_smdp = np.zeros((nS,2), np.float64) # smdp (expected) rewards
        self.tau_bar = np.zeros((nS, 2), np.float64) # Expected holding times

        #Fill out smdp probabilities. Chapman-Kolmogorov equation
        for s in np.arange(nS):
            for a in np.arange(2):
                for t in np.arange(T_max):
                    self.P_smdp[s,a,:] += np.linalg.matrix_power(self.P[:,a,:], t+1)[s,:]*self.tau[:,a,t]
                    self.tau_bar[s,a] += (t+1)*self.tau[s,a,t]
                    self.R_smdp[s,a] += (1-np.sum(self.tau[s,a,:t]))*(np.linalg.matrix_power(self.P[:,a,:], t)@self.R[:,a])[s]

        # Calculate the equivalent MDP 
        self.R_eq = self.R_smdp/self.tau_bar
        for s in np.arange(nS):
            for a in np.arange(2):
                for s_new in np.arange(nS):
                    if s == s_new:
                        delta = 1
                    else:
                        delta = 0
                    self.P_eq[s,a,s_new] = 0.9/self.tau_bar[s,a] * (self.P_smdp[s,a,s_new] - delta) + delta

# To reset the environment in initial settings.
    def reset(self):
        self.s = 0
        return self.s

# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
# Idea is to make an if else statement like before.
    def step(self, action):
        tau = self._rand_choice_nb(np.arange(1,self.T_max+1, dtype=np.int64), self.tau[self.s,action])  # draw holding time uniformly
        reward = 0
        for i in np.arange(1,tau+1):
            new_s = self._rand_choice_nb(np.arange(self.nS, dtype=np.int64), self.P[self.s, action])
            reward += self.R[self.s, action] # get termination reward (will be last)
            self.s = new_s # set state
        return new_s, reward, tau
    
    def _rand_choice_nb(self, arr, prob):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]