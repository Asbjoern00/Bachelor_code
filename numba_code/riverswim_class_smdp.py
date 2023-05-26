# Implementation of riverswim as a pure SMDP
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
    ('distribution', int64),
    ('param', float64),
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
class riverswim():
    """Riwerswim class

    Attributes
    ----------
    P : np.array
        Transition probability matrix of the underlying MDP
    P_smdp : np.array
        Transition probability matrix of the smdp induced on the MDP by holding times
    p_eq : np.array
        The equivalent transition probability matrix of the smdp
    distribution: int
        1=Uniform, 2 = Constant, 3 = Poison, 4 = Binomial, 5 = Geometric
    param: float
        Parameter of the distribution. Only used if distribution is poison, binomial or geometric 

    """
	
    def __init__(self, nS,T_max,distribution, param=1.0):
        self.nS = nS
        self.nA = 2 # two options.
        nS = self.nS
        self.P = np.zeros((nS, 2, nS), np.float64) # Transition probabilities (smdp).
        self.P_eq = np.zeros((nS, 2, nS), np.float64) # Transition probabilities in equivalent MDP.
        
        self.T_max = T_max
        self.distribution = distribution
        self.param = param
        
        # does not matter
        self.tau = np.full((nS, 2, self.T_max), 1/self.T_max, np.float64) # Holding times. For now assumed uniform
        self.tau[:,0,:] = 0
        self.tau[:,0,0] = 1 
 
        

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

        # Calculate expected holding times (known distributions).
        # Generally, we assume that holding time of going left is 1. 
        self.tau_bar = np.ones((nS, 2), np.float64) # Expected holding times (default is 1) for both
        # Uniform holding time of right in (1,T_max) 
        if self.distribution == 1:
            self.tau_bar[:,1] = np.full(nS,(T_max+1)/2, np.float64) # mean of first T_max numbers
        if self.distribution == 2:
            self.tau_bar[:,1] = np.full(nS,T_max, np.float64) # mean of first T_max numbers
        # Poisson distribution (letting 0 be 1)
        if self.distribution == 3:
            self.tau_bar[:,1] = np.full(nS,self.param+np.exp(-self.param), np.float64) # mean of poisson
        # Binomial distribution (0 and 1 drawn as the same)
        if self.distribution == 4:
            self.tau_bar[:,1] = np.full(nS,T_max * self.param + (1-self.param)**T_max, np.float64) # note n is T_max
        # Hypergeometrical distribution (0 and 1 are the same)
        if self.distribution == 5:
            self.tau_bar[:,1] = np.full(nS,1/self.param, np.float64) # mean geometric distribution.

        # Find SMDP matrices. 
        self.R_smdp = self.R
        self.P_smdp = self.P # Transition happens with same probs. 
        # Calculate the equivalent MDP 
        self.R_eq = self.R/self.tau_bar
        for s in np.arange(nS):
            for a in np.arange(2):
                for s_new in np.arange(nS):
                    if s == s_new:
                        delta = 1
                    else:
                        delta = 0
                    self.P_eq[s,a,s_new] = 0.9/self.tau_bar[s,a] * (self.P[s,a,s_new] - delta) + delta

# To reset the environment in initial settings.
    def reset(self):
        self.s = 0
        return self.s

# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
# Idea is to make an if else statement like before.
    def step(self, action):
        # let tau depend on action - long holding times for right.
        if action == 0:
            tau = 1.0
        if action == 1:
            # Uniform
            if self.distribution == 1:
                tau = self._rand_choice_nb(np.arange(1,self.T_max+1,dtype = np.int64),self.tau[self.s,1])  # draw holding time uniformly
            if self.distribution == 2:
                tau = self.T_max  # draw holding time from constant distribution.
            # Poisson distribution (letting 0 be 1)
            if self.distribution == 3:
                tau = np.random.poisson(lam = self.param)  # draw holding time from modified poisson
            # Binomial distribution (0 and 1 drawn as the same)
            if self.distribution == 4:
                tau = np.random.binomial(n = self.T_max, p = self.param)  # draw holding time from modified binomial
            # Hypergeometrical distribution (0 and 1 are the same)
            if self.distribution == 5:
                tau = np.random.geometric(p = self.param)  # draw holding time from geometric distribution
        if tau == 0: # replace tau as 1 if 0 for distribution in talk.
            tau = 1.0
        reward = 0.0
        new_s = self._rand_choice_nb(np.arange(self.nS, dtype=np.int64), self.P[self.s, action])
        reward += self.R[self.s, action] # get termination reward
        self.s = new_s # set state
        return new_s, reward, tau
    
    def _rand_choice_nb(self, arr, prob):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]