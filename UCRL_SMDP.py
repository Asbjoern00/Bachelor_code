import numpy as np
class UCRL_SMDP:
    def __init__(self, nS, nA , delta, b_r, sigma_r, b_tau, sigma_tau, r_max, tau_min, tau_max):
        
        #Assign attributes to instance
        self.nS = nS
        self.nA = nA
        self.delta = delta
        self.b_r = b_r
        self.sigma_r = sigma_r
        self.b_tau = b_tau
        self.sigma_tau = sigma_tau
        self.r_max = r_max
        self.tau_min = tau_min
        self.tau_max = tau_max

        # For speedup of EVI
        self.current_bias_estimate = np.zeros(self.nS)
        # Also for EVI. See Fruits code
        self.tau = self.tau_min-0.1
        

        
        #For detecting last action
        self.last_action = -1

        # The "counter" variables:
        self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        self.tausa = np.zeros((self.nS, self.nA)) # Cumulated holding time observed for (s, a).
        self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.
        
        #Counter variables that might be unnecessary
        self.i = 1
        self.n_episodes = 0 # added

        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        self.hattau = np.zeros((self.nS, self.nA)) # Estimate of holding time

        # Confidence intervals:
        self.confR = np.zeros((self.nS, self.nA))
        self.confP = np.zeros((self.nS, self.nA))
        self.conftau = np.zeros((self.nS, self.nA))
        
        # The current policy (updated at each episode).
        self.policy = np.zeros((self.nS,), dtype=int)
    
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]


    def set_state(self, state):
        self.current_state = state
    
    def new_episode(self):
        self.updateN() # We update the counter Nk.
        self.vk = np.zeros((self.nS, self.nA))
        self.n_episodes +=1 # add to episode counter

        # Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                self.hatR[s, a] = self.Rsa[s, a] / div
                self.hattau[s,a] = self.tausa[s,a] / div
                for next_s in range(self.nS):
                    self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

        # Update the confidence intervals and policy.
        self.confidence()
        self.policy = self.EVI()



    def confidence(self):
        """Computes confidence intervals. See section *Confidence Intervals* in Fruit
        """
        for s in range(self.nS):
            for a in range(self.nA):                                
                n = max(1,self.Nk[s,a])
                #Probability
                self.confP[s,a] = np.sqrt( (14 * self.nS * np.log(2*self.nA*self.i/self.delta) ) / (n) )
                
                #Holding time
                if self.Nk[s,a] >= (2*self.b_tau**2)/(self.sigma_tau**2)*np.log((240*self.nS*self.nA*self.i**7)/ (self.delta)):
                    self.conftau[s,a] = self.sigma_tau * np.sqrt( (14 * np.log(2*self.nS*self.nA*self.i/self.delta) ) / (n))
                else:
                    self.conftau[s,a] = 14 * self.b_tau *  ( np.log(2*self.nS*self.nA*self.i/self.delta) ) / (n)

                #Rewards
                if self.Nk[s,a] >= (2*self.b_r**2)/(self.sigma_r**2)*np.log((240*self.nS*self.nA*self.i**7)/ (self.delta)):
                    self.confR[s,a] = self.sigma_r * np.sqrt( (14 * np.log(2*self.nS*self.nA*self.i/self.delta) ) / (n))
                else:
                    self.confR[s,a] = 14 * self.b_r *  ( np.log(2*self.nS*self.nA*self.i/self.delta) ) / (n)
    
    def max_proba(self, sorted_indices, s, a):
        """Maximizes over probability distribution in confidence set

        Parameters
        ----------
        sorted_indices : np.array
            sorted (smallest to largest) indices 
        s : int
            integer representation of state
        a : int
            integer representation of action

        Returns
        -------
        max_p: np.array
            maximizing probability distribution 
        """
		
        min1 = min([1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2)])
        max_p = np.zeros(self.nS)
            
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = np.copy(self.hatP[s, a])
            max_p[sorted_indices[-1]] += self.confP[s, a] / 2
            l = 0 
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
                l += 1        
        return max_p
    
    # The Extended Value Iteration, perform an optimisitc VI over a set of MDP.
	#Note, changed fixed epsilon to 1/sqrt(i)
    def EVI(self, max_iter = 2*10**3):
        """Does EVI on extended by converting SMDP to equivalent extended MDP

        Parameters
        ----------
        max_iter : int, optional
            Max iteration to run EVI for, by default 2*10**3

        Returns
        -------
        policy : np.array
            Optimal policy w.r.t. optimistic SMDP 
        """
        niter = 0
        epsilon = self.r_max/np.sqrt(self.i)
        sorted_indices = np.arange(self.nS)
        action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]

        # The variable containing the optimistic policy estimate at the current iteration.
        policy = np.zeros(self.nS, dtype=int)

        # Initialise the value and epsilon as proposed in the course.
        V0 = self.current_bias_estimate # NB: setting it to the bias obtained at the last episode can help speeding up the convergence significantly!, Done!
        V1 = np.zeros(self.nS)
        r_tilde = np.zeros((self.nS, self.nA))
        tau_tilde = np.zeros((self.nS, self.nA))
        # The main loop of the Value Iteration algorithm.
        while True:
            niter += 1
            for s in range(self.nS):
                for a in range(self.nA):
                    maxp = self.max_proba(sorted_indices, s, a)

                    r_tilde[s,a] = min(self.hatR[s,a] + self.confR[s,a], self.r_max*self.tau_max)
                    
                    tau_tilde[s,a] = min(self.tau_max, max(self.tau_min, self.hattau[s,a] - np.sign(r_tilde[s,a] +  self.tau*((maxp.T @ V0)-V0[s])*self.conftau[s,a])))
                    
                    temp = r_tilde[s,a]/tau_tilde[s,a]+(self.tau/tau_tilde[s,a])*((maxp.T@V0) - V0[s])+V0[s]
                    if (a == 0) or ((temp + action_noise[a]) > (V1[s] + action_noise[self.policy[s]])): # Using a noise to randomize the choice when equals.
                        V1[s] = temp
                        policy[s] = a

            # Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
            diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
            if (max(diff) - min(diff)) < epsilon:
                self.current_bias_estimate = V1
                return policy
            else:
                V0 = V1
                V1 = np.zeros(self.nS)
                sorted_indices = np.argsort(V0)
            if niter > max_iter:
                print("No convergence in EVI after: ", max_iter, " steps!", maxp)
                return policy
    
    def play(self, state, reward,tau):

        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
            self.tausa[self.s, self.last_action] += tau

        action = self.policy[state]
        if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
            self.new_episode()
            action  = self.policy[state]

        # Update the variables:
        self.vk[state, action] += 1
        #self.obtained_rewards[self.s, self.last_action, self.t-1] = reward
        self.s = state
        self.last_action = action
        self.i += 1

        return action, self.policy

    def reset(self, init=0):
            # The "counter" variables:
            self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
            self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
            self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
            self.tausa = np.zeros((self.nS, self.nA))
            self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.
            
            # The "estimates" variables:
            self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
            self.hatR = np.zeros((self.nS, self.nA))
            
            # Confidence intervals:
            self.confR = np.zeros((self.nS, self.nA))
            self.confP = np.zeros((self.nS, self.nA))

            # The current policy (updated at each episode).
            self.policy = np.zeros((self.nS,), dtype=int)

            # Set the initial state and last action:
            self.s = init
            self.last_action = -1
            self.i = 1

            #self.obtained_rewards = np.empty((self.nS,self.nA,2*10**6))
            
            # Start the first episode.
            self.new_episode()
