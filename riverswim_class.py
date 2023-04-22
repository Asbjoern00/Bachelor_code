# Implementation of riverswim as an SMDP (/MDP with options)
import numpy as np
# We let ourselves be inspired by some of the provided source code from the course OReL.
# But we modify it to handle options. 


class riverswim():
	
    def __init__(self, nS,T_max):
        self.nS = nS
        self.nA = 2 # two options.
        nS = self.nS
        self.P = np.zeros((nS, 2, nS)) # Transition probabilities.
        self.P_op = np.zeros((nS, 2, nS)) # Transition probabilities.
        self.T_max = T_max
        self.tau = np.zeros((nS, 2, nS)) # Holding times.
        self.tau_bar = np.zeros((nS,2)) # expected holding time.
        self.P_eq = np.zeros((nS,2,nS)) # Equivalent probabilities. 
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


        self.P_eq = self.P # this naming is necessary for now.

# We build the reward matrix R (same as simple implementation)
        self.R = np.zeros((nS, 2))
        self.R[0, 0] = 0.05
        self.R[nS - 1, 1] = 1
        # We (arbitrarily) set the initial state in the leftmost position.
        self.s = 0

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

        '''elif self.s==self.nS-1 and action==1:
            tau=1
        elif self.nS-self.s>self.T_max and action==1:
            tau = np.random.choice(range(1,self.T_max+1),replace = True)  # draw holding time uniformly i.e. term prob.
        elif self.nS-self.s<=self.T_max and action==1:
            tau = np.random.choice(range(1,self.nS-self.s+1),replace = True)  # draw holding time uniformly i.e. term prob.
            '''

        for i in range(1,tau+1):
            new_s = np.random.choice(np.arange(self.nS), p=self.P_eq[self.s, action])
            reward = self.R[self.s, action] # get termination reward (will be last)
            self.s = new_s # get termination reward (will be last)
        return new_s, reward, tau

# Note we need to add equivalent prob measures too.
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