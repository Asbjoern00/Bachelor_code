import numpy as np
# PURPOSE: IMPLEMENTATION OF EPSILON-GREEDY ALGORITHM.
# So far to look like Sadeghs slides on online average reward RL (14/49) 
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
            if (max(diff) - min(diff)) < self.eps():
                return policy
            else:
                V0 = V1
                V1 = np.zeros(env.nS)
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
		
        policy = self.VI() # np.argmax(self.hatR,axis = 1)
        dum = np.random.choice(2,replace=True,p = [self.eps(),1-self.eps()])
        if dum ==1 : # i.e. greedily chosen:
            action = policy[state]            
        else:
            action = np.random.choice(self.nA,replace = True)
        tau = 1
		# Update the variables:
        self.Nsa[state, action] += 1
        self.s = state
        self.last_action = action
        self.i += 1 
        print(self.i)
        return action, self.policy
    def reset(self, init):
		# The "counter" variables:
        self.Nsa = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a).
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).

		# The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix
        self.hatR = np.zeros((self.nS, self.nA))
		
		# Set the initial state and last action:
        self.s = init
        self.last_action = -1
        



