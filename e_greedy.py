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
        self.tausa = np.ones((self.nS, self.nA)) # Cumulated holding time observed for (s, a).
        self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

        #Counter variables that might be unnecessary
        self.i = 1
        #For detecting last action
        self.last_action = -1 # initialize with this.
        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hatR_eq = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hattau = np.ones((self.nS, self.nA)) # Estimate of tau's. Assume 1 unless other stated.
        self.hatP_eq = np.zeros((self.nS, self.nA, self.nS)) 
        # The current policy (updated at each decision step).
        self.policy = np.zeros((self.nS,), dtype=int)
        self.n_episodes = 0 # add to episode counter

    def eps(self):
        return 1/np.sqrt(self.n_episodes) # return 1/sqrt(episode step)
    
    def VI(self,max_iter = 10**2):

        # The variable containing the optimal policy estimate at the current iteration.
        policy = np.zeros(self.nS, dtype=int)
        niter = 0

        # Initialise the value and epsilon as proposed in the course.
        V0 = np.zeros(self.nS)
        V1 = np.zeros(self.nS)
        Q1 = np.zeros((self.nS,self.nA))
        # The main loop of the Value Iteration algorithm.
        while True:
            niter += 1
            # The below is a way of breaking arbitrarily. 
            for s in range(self.nS):
                for a in range(self.nA): 
                    Q1[s,a] = self.hatR_eq[s,a] + V0 @ self.hatP_eq[s,a,:]  # Note Peq instead of P

            index = Q1 == np.max(Q1,axis=1).reshape((self.nS,1))
            probs = index/np.sum(index,axis=1).reshape((self.nS,1))
            # Do this sample for all (if uniue sampled w.p. 1)
            for i in range(0,self.nS):
                values = range(0,self.nA)
                policy[i]=np.random.choice(values,replace = True,p=probs[i,:])
            V1 = np.max(Q1,axis=1) # only value is important.
            # Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
            gain = 0.5*(max(V1 - V0) + min(V1 - V0))
            diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
            if (max(diff) - min(diff)) < self.eps():
                return policy
            else:
                V0 = V1
                V1 = np.zeros(self.nS)
            if niter > max_iter:
                print("No convergence in VI after: ", max_iter, " steps!")
                return policy

    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nsa[s, a] += self.vk[s, a]

    def new_episode(self):
        self.updateN() # We update the counter Nk.
        self.vk = np.zeros((self.nS, self.nA))
        self.n_episodes +=1 # add to episode counter

        # Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
		# Update estimates and confidence intervals.
        for s in range(self.nS):
            for a in range(self.nA):
                    self.hatR[s, a] = self.Rsa[s, a] / max((1, self.Nsa[s, a]))
                    self.hattau[s, a] = self.tausa[s, a] / max((1, self.Nsa[s, a]))

                    for ss in range (self.nS):
                        self.hatP[s, a, ss] = self.Nsas[s, a, ss] / max((1, self.Nsa[s, a]))
        self.hatR_eq = self.hatR/self.hattau
        # Update equivalent measures. 
        for s in range(self.nS):
            for a in range(2):
                for s_new in range(self.nS):
                    if s == s_new:
                        delta = 1
                    else:
                        delta = 0
                    self.hatP_eq[s,a,s_new] = 0.9/self.hattau[s,a] * (self.hatP[s,a,s_new] - delta) + delta 


        self.policy = self.VI() # Update policy by value iteration. 


    def play(self,state,reward,tau): # here just time step/decision step.
        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
            self.tausa[self.s, self.last_action] += tau

        # New action - draw epsiolon greedily:
        dum = np.random.choice(2,replace=True,p = [1/self.i,1-1/self.i]) # draw each time step w.p. 1/ decision step
        action = self.policy[state]            

        if dum == 0: # i.e. greedily chosen:
            action = np.random.choice(self.nA,replace = True)

		# Update the variables:

        if self.vk[state, action] > max([1, self.Nsa[state, action]]): # Stoppping criterion
            self.episode_ended = True
            self.new_episode()


        # Update the variables:
        self.vk[state, action] += 1
        self.s = state
        self.last_action = action
        self.i += 1

        return action, self.policy
    
    def reset(self, init):
		# The "counter" variables:
        self.Nsa = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a).
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        self.tausa = np.ones((self.nS, self.nA)) # Cumulated holding time observed for (s, a).


		# The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hatR_eq = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hattau = np.ones((self.nS, self.nA)) # Estimate of holding time
        self.hatP_eq = np.zeros((self.nS, self.nA, self.nS)) 
		
		# Set the initial state and last action:
        self.s = init
        self.last_action = 0
        

'''
class GREEDY():
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        # The "counter" variables:
        self.Nsa = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        self.tausa = np.ones((self.nS, self.nA)) # Cumulated holding time observed for (s, a).

        #Counter variables that might be unnecessary
        self.i = 1
        #For detecting last action
        self.last_action = 0 # initialize with this.
        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hatR_eq = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hattau = np.ones((self.nS, self.nA)) # Estimate of tau's. Assume 1 unless other stated.
        self.hatP_eq = np.zeros((self.nS, self.nA, self.nS)) 
        # The current policy (updated at each decision step).
        self.policy = np.zeros((self.nS,), dtype=int)
    
    def eps(self):
        return 1/np.sqrt(self.i) # return 1/sqrt(decision step)
    
    def VI(self,max_iter = 10**2):

        # The variable containing the optimal policy estimate at the current iteration.
        policy = np.zeros(self.nS, dtype=int)
        niter = 0

        # Initialise the value and epsilon as proposed in the course.
        V0 = np.zeros(self.nS)
        V1 = np.zeros(self.nS)
        Q1 = np.zeros((self.nS,self.nA))
        # The main loop of the Value Iteration algorithm.
        while True:
            niter += 1
            # The below is a way of breaking arbitrarily. 
            for s in range(self.nS):
                for a in range(self.nA): 
                    Q1[s,a] = self.hatR_eq[s,a] + V0 @ self.hatP_eq[s,a,:]  # Note Peq instead of P

            index = Q1 == np.max(Q1,axis=1).reshape((self.nS,1))
            probs = index/np.sum(index,axis=1).reshape((self.nS,1))
            # Do this sample for all (if uniue sampled w.p. 1)
            for i in range(0,self.nS):
                values = range(0,self.nA)
                policy[i]=np.random.choice(values,replace = True,p=probs[i,:])
            V1 = np.max(Q1,axis=1) # only value is important.
            # Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
            gain = 0.5*(max(V1 - V0) + min(V1 - V0))
            diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
            if (max(diff) - min(diff)) < self.eps():
                return policy
            else:
                V0 = V1
                V1 = np.zeros(self.nS)
            if niter > max_iter:
                print("No convergence in VI after: ", max_iter, " steps!")
                return policy

    def play(self,state,reward,tau): # here just time step/decision step.
        self.Nsas[self.s, self.last_action, state] += 1
        self.Rsa[self.s, self.last_action] += reward
        self.tausa[self.s, self.last_action] += tau



		# Update estimates and confidence intervals.
        for s in range(self.nS):
            for a in range(self.nA):
                    self.hatR[s, a] = self.Rsa[s, a] / max((1, self.Nsa[s, a]))
                    self.hattau[s, a] = self.tausa[s, a] / max((1, self.Nsa[s, a]))

                    for ss in range (self.nS):
                        self.hatP[s, a, ss] = self.Nsas[s, a, ss] / max((1, self.Nsa[s, a]))
        self.hatR_eq = self.hatR/self.hattau
        # Update equivalent measures. 
        for s in range(self.nS):
            for a in range(2):
                for s_new in range(self.nS):
                    if s == s_new:
                        delta = 1
                    else:
                        delta = 0
                    self.hatP_eq[s,a,s_new] = 0.9/self.hattau[s,a] * (self.hatP[s,a,s_new] - delta) + delta      
        policy = self.VI() # np.argmax(self.hatR,axis = 1)
        dum = np.random.choice(2,replace=True,p = [self.eps(),1-self.eps()])
        if dum ==1 : # i.e. greedily chosen:
            action = policy[state]            
        else:
            action = np.random.choice(self.nA,replace = True)
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
        self.tausa = np.ones((self.nS, self.nA)) # Cumulated holding time observed for (s, a).


		# The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hatR_eq = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hattau = np.ones((self.nS, self.nA)) # Estimate of holding time
        self.hatP_eq = np.zeros((self.nS, self.nA, self.nS)) 
		
		# Set the initial state and last action:
        self.s = init
        self.last_action = 0
        
'''

