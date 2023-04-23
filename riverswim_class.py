# Implementation of riverswim as an SMDP (/MDP with options)
import numpy as np
# We let ourselves be inspired by some of the provided source code from the course OReL.
# But we modify it to handle options. 


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

    """
	
    def __init__(self, nS,T_max):
        self.nS = nS
        self.nA = 2 # two options.
        nS = self.nS
        self.P = np.zeros((nS, 2, nS)) # Transition probabilities.
        self.P_eq = np.zeros((nS, 2, nS)) # Transition probabilities in equivalent MDP.
        
        self.T_max = T_max
        
        self.tau = np.full((nS, 2, self.T_max), 1/self.T_max) # Holding times. For now assumed uniform
 
        
        self.beta = np.full(nS,1/nS) # uniform termination prob. 

        # action probs.
        for s in range(nS):
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
        self.R = np.zeros((nS, 2))
        self.R[0, 0] = 0.05
        self.R[nS - 1, 1] = 1
        # We (arbitrarily) set the initial state in the leftmost position.
        self.s = 0

        self.P_smdp = np.zeros((nS,2,nS)) # smdp probabilities.
        self.R_smdp = np.zeros((nS,2)) # smdp (expected) rewards
        self.tau_bar = np.zeros((nS, 2)) # Expected holding times

        #Fill out smdp probabilities. Chapman-Kolmogorov equation
        for s in range(nS):
            for a in range(2):
                for t in range(T_max):
                    self.P_smdp[s,a,:] += np.linalg.matrix_power(self.P[:,a,:], t+1)[s,:]*self.tau[:,a,t]
                    self.tau_bar[s,a] += (t+1)*self.tau[s,a,t]
                    self.R_smdp[s,a] += (1-np.sum(self.tau[s,a,:t]))*(np.linalg.matrix_power(self.P[:,a,:], t)@self.R[:,a])[s]

        # Calculate the equivalent MDP 
        self.R_eq = self.R_smdp/self.tau_bar
        for s in range(nS):
            for a in range(2):
                for s_new in range(nS):
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
        if action==0: #always holding time of 1 for left.
            tau=1
        else:
            tau = np.random.choice(range(1,self.T_max+1),replace = True)  # draw holding time uniformly i.e. term prob.
        reward = 0
        for i in range(1,tau+1):
            new_s = np.random.choice(np.arange(self.nS), p=self.P_eq[self.s, action])
            reward += self.R[self.s, action] # get termination reward (will be last)
            self.s = new_s # get termination reward (will be last)
        return new_s, reward, tau





















'''
env = riverswim(nS = 10, T_max = 9)

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

print(QL_SMDP(env,T=10**5))
'''