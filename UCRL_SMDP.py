import numpy as np
class UCRL_SMDP:
    def __init__(self, nS, nA , delta=0.05, b_r=1, sigma_r=1/2, b_tau=1, r_max=1, tau_min=1, sigma_tau=None, tau_max=None, T_max = None):
        
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
        self.T_max = T_max
        self.episode_ended = False

        if self.tau_max is None and self.sigma_tau is None and self.T_max is not None:
            self.tau_max = self.T_max
            self.sigma_tau = max(1,(self.T_max-1)/2) # Assuming bounded holding and a minimum holding time of 1            

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
    
    def set_state(self, state):
        self.s = state


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
    def EVI(self, max_iter = 10**3):
        """Does EVI on extended by converting SMDP to equivalent extended MDP

        Parameters
        ----------
        max_iter : int, optional
            Max iteration to run EVI for, by default 10**3

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
                print(f"No convergence in EVI after: , {max_iter},  steps!. Actual diff was {max(diff)-min(diff)}, and epsilon = {epsilon}"  )
                return policy
    
    def play(self, state, reward, tau):

        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
            self.tausa[self.s, self.last_action] += tau

        action = self.policy[state]
        if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
            self.episode_ended = True
            self.new_episode()
            action  = self.policy[state]

        # Update the variables:
        self.vk[state, action] += 1
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
            
            # Start the first episode.
            self.new_episode()

class BUS(UCRL_SMDP):
    def __init__(self, nS, nA, delta, b_r, sigma_r, b_tau, sigma_tau, r_max, tau_min, tau_max, T_max_grid):
        self.T_max_grid = T_max_grid
        self.loss_grid = np.zeros(len(T_max_grid)) # For sampling the algorithms
        self.current_sample_prop = np.ones(len(T_max_grid))/len(T_max_grid)
        super().__init__(nS, nA, delta, b_r, sigma_r, b_tau, sigma_tau, r_max, tau_min, tau_max)
        
        self.sample_parameters()
        self.update_parameters()
        
    
    def learning_rate(self):
        return np.sqrt(np.log(len(self.T_max_grid))/(self.n_episodes*len(self.T_max_grid)))

    def sample_prob(self):
        numerator = np.exp(-self.learning_rate()*(self.loss_grid-np.min(self.loss_grid)))
        self.numerator = numerator
        self.current_sample_prop = numerator/np.sum(numerator)

    def update_parameters(self):
        self.tau_max = self.current_T_max
        self.sigma_tau = max(1,(self.current_T_max-1)/2) # Assuming bounded holding and a minimum holding time of 1
    
    def sample_parameters(self):
        self.current_T_max = np.random.choice(self.T_max_grid, size = 1, p = self.current_sample_prop)
        self.current_T_max_index = np.where(self.current_T_max == self.T_max_grid)[0]

    def play(self, state, reward, tau, sample_prob):

        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
            self.tausa[self.s, self.last_action] += tau

            self.current_episode_loss += (self.r_max - reward)/(sample_prob) # Added for compatibility w. BUS-like algos

        action = self.policy[state]
        if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
            self.loss_grid[self.current_T_max_index] += self.current_episode_loss/np.sum(self.vk) # Update loss with average importance weighted loss of the episode
            self.new_episode()
            action  = self.policy[state]

        # Update the variables:
        self.vk[state, action] += 1
        self.s = state
        self.last_action = action
        self.i += 1

        return action, self.policy


    def new_episode(self):
        self.updateN() # We update the counter Nk.
        self.vk = np.zeros((self.nS, self.nA))
        self.current_episode_loss = 0
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
        self.sample_prob()
        self.sample_parameters()
        self.update_parameters() 
        self.confidence()
        self.policy = self.EVI()

class BUS2():
    def __init__(self, nS, nA ,T_max_grid, delta=0.05, b_r=1, sigma_r=1/2, b_tau=1, r_max=1, tau_min = 1):
        self.nS = nS
        self.nA = nA
        self.delta = delta
        self.b_r = b_r
        self.sigma_r = sigma_r
        self.b_tau = b_tau
        self.r_max = r_max
        self.tau_min = tau_min
        self.T_max_grid = T_max_grid

        self.loss_grid = np.zeros(len(T_max_grid)) # For sampling the algorithms
        self.current_sample_prop = np.ones(len(T_max_grid))/len(T_max_grid)
        
        self.algorithms = [UCRL_SMDP(nS = self.nS, nA = self.nA, delta = self.delta, b_r = self.b_r, sigma_r=self.sigma_r, b_tau=self.b_tau, tau_min=self.tau_min, T_max=t)
                            for t in T_max_grid]
        
        self.sample_parameters()


    def learning_rate(self):
        self.n_episodes = sum([self.algorithms[i].n_episodes for i in range(len(self.T_max_grid))])
        return np.sqrt(np.log(len(self.T_max_grid))/(self.n_episodes*len(self.T_max_grid)))

    def sample_prob(self):
        numerator = np.exp(-self.learning_rate()*(self.loss_grid-np.min(self.loss_grid)))
        self.current_sample_prop = numerator/np.sum(numerator)
    
    def sample_parameters(self):
        self.current_episode_loss = 0 
        self.current_T_max = np.random.choice(self.T_max_grid, size = 1, p = self.current_sample_prop)
        self.current_T_max_index = np.where(self.current_T_max == self.T_max_grid)[0]
        self.current_algorithm = self.algorithms[int(self.current_T_max_index)]
    
    def reset(self, s):
        self.s = 0 
        self.loss_grid = np.zeros(len(self.T_max_grid)) # For sampling the algorithms
        self.current_sample_prop = np.ones(len(self.T_max_grid))/len(self.T_max_grid)

        for i in range(len(self.T_max_grid)):
            self.algorithms[i].reset()


    def play(self, state, reward, tau):

        if self.current_algorithm.episode_ended:
            self.current_algorithm.episode_ended = False
            state = self.current_algorithm.s # store the current state so can be transfered to the next sampled algo
            self.loss_grid[self.current_T_max_index] += (self.current_episode_loss/self.current_sample_prop[self.current_T_max_index])*1/self.current_episode_length
            self.sample_prob() # calculate new distribution
            self.sample_parameters() # do sampling
            self.current_algorithm.s = state # update state of current algo 
            self.play(state, reward, tau) # play current algo
        

        self.current_episode_loss += (self.r_max - reward) # add loss for current episode
        self.current_episode_length = np.sum(self.current_algorithm.vk) # find length of episode
        action, policy = self.current_algorithm.play(state, reward, tau) # play current algo

        
            
        return action,policy

