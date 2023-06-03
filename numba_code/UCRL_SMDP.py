import numpy as np
from numba import int64, float64,boolean
from numba.experimental import jitclass
from numba import types, typed, typeof

spec = [
	('nS', int64),
	('T_max', int64),
	('nA', int64),
	('delta', float64),
    ('b_r', float64),
    ('b_tau', float64),
    ('r_max', float64),
    ('tau_min', float64),
    ('imprv', int64),
    ('sigma_r', float64),
	('sigma_tau', float64),
    ('tau_max', float64),
    ('T_max', float64),
    ('episode_ended',boolean),
    ('current_bias_estimate', float64[:]),
    ('n_episodes',int64),
    ('s', int64),
	('tau', float64),
    ('Nk',int64[:,:]),
    ('Nsas', int64[:,:,:]),
    ('Rsa',float64[:,:]),
    ('vk', int64[:,:]),
    ('tausa', float64[:,:]),
    ('hatR', float64[:,:]),
	('hatP', float64[:,:,:]),
    ('hattau', float64[:,:]),
    ('confR', float64[:,:]),
	('confP', float64[:,:]),
    ('conftau', float64[:,:]),
    ('policy', int64[:]),
    ('i', int64),
    ('last_action', int64)
    ]



@jitclass(spec=spec)
class UCRL_SMDP:
    def __init__(self, nS, nA, delta=0.05, b_r=1, b_tau=1, r_max=1, tau_min=1,imprv=0, sigma_r=-1, sigma_tau=-1, tau_max=-1, T_max = 1):
        
        #Assign attributes to instance
        self.nS = nS
        self.nA = nA
        self.delta = delta
        self.b_r = b_r
        self.b_tau = b_tau
        self.sigma_tau = sigma_tau
        self.r_max = r_max

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.T_max = T_max
        self.sigma_r = sigma_r
        self.imprv = imprv
        if self.imprv == 1:
            self.delta = 1/6*self.delta # need to readjust delta to make algorithms comparable. See proof of smdp-ucrl-l regret bound
        self.episode_ended = False

        if (self.tau_max < 0 ) and (self.sigma_r < 0) and (self.sigma_tau  < 0 ) and (self.T_max is not None):
            self.tau_max = self.T_max
            self.sigma_tau = (self.T_max-1)/2 # Assuming bounded holding and a minimum holding time of 1 
            self.sigma_r = self.r_max*self.tau_max/2 # Assuming bounded holding and a minimum holding time of 1 
        
        

        # For speedup of EVI
        self.current_bias_estimate = np.zeros(self.nS, dtype=np.float64)
        # Also for EVI. See Fruits code
        self.tau = self.tau_min-0.1
        

        
        #For detecting last action
        self.last_action = -1

        # The "counter" variables:
        self.Nk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated reward observed for (s, a).
        self.tausa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated holding time observed for (s, a).
        self.vk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) in the current episode.
        
        #Counter variables that might be unnecessary
        self.i = 1
        self.n_episodes = 0 # added

        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA), dtype=np.float64) # Estimate of rewards
        self.hattau = np.zeros((self.nS, self.nA), dtype=np.float64) # Estimate of holding time

        # Confidence intervals:
        self.confR = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.confP = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.conftau = np.zeros((self.nS, self.nA), dtype=np.float64)
        
        # The current policy (updated at each episode).
        self.policy = np.zeros(self.nS, dtype=np.int64)
    
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]


    def new_episode(self):
        self.updateN() # We update the counter Nk.
        self.vk = np.zeros((self.nS, self.nA), dtype = np.int64)
        self.n_episodes +=1 # add to episode counter

        # Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
        for s in range(self.nS):
            for a in range(self.nA):
                div = np.max(np.array([1, self.Nk[s, a]], dtype=np.float64))
                self.hatR[s, a] = self.Rsa[s, a] / div
                self.hattau[s,a] = self.tausa[s,a] / div
                for next_s in range(self.nS):
                    self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

        # Update the confidence intervals and policy.
        self.confidence()
        self.policy = self.EVI()
        #print(f"New episode started at {self.i}")
    
    def set_state(self, state):
        self.s = state


    def confidence(self):
        """Computes confidence intervals. See section *Confidence Intervals* in Fruit
        Parameters: 
         - imprv: Takes on values 0,1,2. These resemble different confidence sets. 
        """
        if self.imprv == 0:
            for s in range(self.nS):
                for a in range(self.nA):                                
                    n = np.max(np.array([1, self.Nk[s, a]], dtype=np.float64))
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
        """Computes improvesconfidence intervals. See Sadegh's note
        """
        if self.imprv == 1: # improved confidence.
            for s in np.arange(self.nS):
                for a in np.arange(self.nA):                                
                    n = np.max(np.array([1, self.Nk[s, a]], dtype=np.float64))
                    #Probability
                    #self.confP[s,a] = np.sqrt( (2*(1+1/n) * np.log(np.sqrt(n+1)*self.nS*self.nA*(2**(self.nS)-2)/self.delta) ) / (n) ) # obs, integer overflow
                    self.confP[s,a] = np.sqrt( (2*(1+1/n) * (self.nS*np.log(2) + np.log(np.sqrt(n+1)*self.nS*self.nA/self.delta) ) / (n) ))


                    #Holding time
                    self.conftau[s,a] = self.sigma_tau * np.sqrt( (2 * (1+1/n) * np.log(self.nS*self.nA*np.sqrt(n+1)/self.delta) ) / (n))

                    #Rewards
                    self.confR[s,a] = self.sigma_r * np.sqrt( (2 * (1+1/n) * np.log(self.nS*self.nA*np.sqrt(n+1)/self.delta) ) / (n))
        """Computes improved confidence intervals. See Brunskill (Only P changes)
        """
        if self.imprv == 2: # improved confidence.
            for s in range(self.nS):
                for a in range(self.nA):                                
                    n = np.max(np.array([1, self.Nk[s, a]], dtype=np.float64))
                    #Probability
                    self.confP[s,a] = np.sqrt( 4 * (2*np.log(np.log(np.max(np.array([n,np.exp(1)]))))+np.log(3*(2**(self.nS)-2)/self.delta)) / (n) )
                    
                    #Holding time
                    self.conftau[s,a] = self.sigma_tau * np.sqrt( (2 * (1+1/n) *self.nS*self.nA* np.log(np.sqrt(n+1)/self.delta) ) / (n))

                    #Rewards
                    self.confR[s,a] = self.sigma_r * np.sqrt( (2 * (1+1/n) *self.nS*self.nA* np.log(np.sqrt(n+1)/self.delta) ) / (n))

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
		
        min1 = np.min(np.array([1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2)]))
        max_p = np.zeros(self.nS, dtype=np.float64)
            
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = np.copy(self.hatP[s, a])
            max_p[sorted_indices[-1]] += self.confP[s, a] / 2
            l = 0 
            while np.sum(max_p) > 1:
                max_p[sorted_indices[l]] = np.max(np.array([0, 1 - np.sum(max_p) + max_p[sorted_indices[l]]]))
                l += 1        
        return max_p
    
    # The Extended Value Iteration, perform an optimisitc VI over a set of MDP.
	#Note, changed fixed epsilon to 1/sqrt(i)

    def EVI(self, max_iter = 10**4):
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
        #epsilon = 0.01
        sorted_indices = np.arange(self.nS, dtype=np.int64)

        # Initialise the value and epsilon as proposed in the course.
        V0 = self.current_bias_estimate # NB: setting it to the bias obtained at the last episode can help speeding up the convergence significantly!, Done!
        V1 = np.zeros(self.nS, np.float64)

        r_tilde = np.minimum(self.hatR + self.confR, self.r_max*self.tau_max) # only needs to be computed once
        # The main loop of the Value Iteration algorithm.
        while True:
            niter += 1
            V0 = np.ascontiguousarray(V0)
            mat_products = np.zeros((self.nS, self.nA))
            tau_tilde = np.zeros((self.nS, self.nA))  
            
            for s in range(self.nS):
                for a in range(self.nA):
                    maxp = self.max_proba(sorted_indices, s, a).T
                    mat_products[s,a] += maxp@V0
                    tau_tilde[s,a] += min(self.tau_max,max(self.tau_min,
                                                self.hattau[s,a] - np.sign(r_tilde[s,a] +  self.tau*((mat_products[s,a]-V0[s]))*self.conftau[s,a])))
                
                V1[s] += np.max(r_tilde[s,:]/tau_tilde[s,:]+(self.tau/tau_tilde[s,:])*((mat_products[s,:]) - V0[s])+V0[s])
            
            diff  = np.abs(V1-V0)
            if (np.max(diff) - np.min(diff)) < epsilon:
                policy = np.zeros(self.nS, dtype=np.int64)
                for s in range(self.nS):
                    candidate_actions = [] # Randomize choices in case of multiple optimal
                    for a in range(self.nA):
                        if np.isclose(V1[s] ,r_tilde[s,a]/tau_tilde[s,a]+(self.tau/tau_tilde[s,a])*((mat_products[s,a]) - V0[s])+V0[s]):
                            #policy[s] = a
                            candidate_actions.append(a)
                        if a == self.nA -1:
                            policy[s] = np.random.choice(np.array(candidate_actions, dtype=np.int64))
                            self.current_bias_estimate = (V1 - np.mean(V1))/(np.std(V1) + 1) # rescale to 0 mean and almost unit variance
                return policy
            
            else:
                V0 = V1
                V1 = np.zeros(self.nS, dtype=np.float64)
                sorted_indices = np.argsort(V0)

            if niter > max_iter:
                print("No convergence in EVI after:" ,max_iter,  "steps!. Actual diff was", max(diff)-min(diff), "and epsilon =", epsilon, 
                      "current bias estimate =" , self.current_bias_estimate)
                policy = np.zeros(self.nS, dtype=np.int64)
                for s in np.arange(self.nS):
                        self.current_bias_estimate = (V1 - np.mean(V1))/(np.std(V1) + 1)
                        policy[s] = np.argmax(r_tilde[s,:]/tau_tilde[s,:]+(self.tau/tau_tilde[s,:])*((mat_products[s,:]) - V0[s])+V0[s])
                return policy
    
    def play(self, state, reward, tau):

        if self.last_action >= 0: # Update if not first action.
            self.Nsas[self.s, self.last_action, state] += 1
            self.Rsa[self.s, self.last_action] += reward
            self.tausa[self.s, self.last_action] += tau

        action = self.policy[state]
        if self.vk[state, action] > np.max(np.array([1, self.Nk[state, action]])): # Stoppping criterion
            #self.episode_ended = True this is still not necessary?
            self.new_episode()
            action  = self.policy[state]

        # Update the variables:
        self.vk[state, action] += 1
        self.s = state
        self.last_action = action
        self.i += 1

        return action, self.policy

    def reset(self,init):
        #For detecting last action
        self.last_action = -1

        # The "counter" variables:
        self.Nk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated reward observed for (s, a).
        self.tausa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated holding time observed for (s, a).
        self.vk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) in the current episode.
        
        #Counter variables that might be unnecessary
        self.i = 1
        self.n_episodes = 0 # added

        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA), dtype=np.float64) # Estimate of rewards
        self.hattau = np.zeros((self.nS, self.nA), dtype=np.float64) # Estimate of holding time

        # Confidence intervals:
        self.confR = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.confP = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.conftau = np.zeros((self.nS, self.nA), dtype=np.float64)
        
        # The current policy (updated at each episode).
        self.policy = np.zeros(self.nS, dtype=np.int64)

        self.s = init 

        # Start the first episode.
        self.new_episode()





spec = [
	('nS', int64),
	('T_max', int64),
	('nA', int64),
	('delta', float64),
    ('r_max', float64),
    ('imprv', int64),
	('sigma_tau', float64),
    ('tau_max', float64),
    ('current_T_max_index', int64),
    ('current_T_max',int64),
    ('n_algos',int64),
    ('play_times', int64[:]),
    ('n_episodes',int64),
    ('s', int64),
	('tau', float64),
    ('Nk',int64[:,:]),
    ('Nsas', int64[:,:,:]),
    ('Rsa',float64[:,:]),
    ('vk', int64[:,:]),
    ('tausa', float64[:,:]),
    ('policy', int64[:]),
    ('i', int64),
    ('last_action', int64),
    ('T_max_grid', int64[:]),
    ('reward_grid', float64[:]),
    ('reward_zero_one', float64[:]),
    ('current_sample_prop', float64[:]),
    ('current_episode_reward', float64),
    ('current_algorithm',UCRL_SMDP.class_type.instance_type),
    ('partition', int64),
    ('H',int64),
    ('algorithms',types.ListType(typeof(UCRL_SMDP(1,1)))), 
    ('eta',float64),
    ('beta',float64),
    ('gamma',float64),
    ('i',int64),
    ('hist_probs', float64[:,:]),
    ('t', int64)
    ]

@jitclass(spec = spec)
class BUS3:
    def __init__(self, nS, nA , H ,delta,imprv,T_max_grid,algorithms):
        self.algorithms = algorithms 
        self.nS = nS
        self.nA = nA
        self.delta = delta/3
        self.imprv = imprv
        self.T_max_grid = T_max_grid
        self.n_algos = T_max_grid.shape[0]
        self.reward_grid = np.zeros(self.n_algos,np.float64) # For sampling the algorithms
        self.reward_zero_one = np.zeros(self.n_algos,np.float64)
        self.current_sample_prop = np.ones(self.n_algos,np.float64)/self.n_algos
        self.partition = 1
        self.H = H
        self.i = 0
        self.t = 0 
        self.hist_probs = np.zeros((10**7, self.n_algos)) # Create large array to populate historic sample proportions

        # Define Paramters as in EXP3.P (but anytime version)
        self.eta = 0.95 * np.sqrt(np.log(self.n_algos) / (self.partition * self.n_algos))
        self.beta = np.sqrt(np.log(self.n_algos/ self.delta) / (self.partition * self.n_algos)) # delta = 0.05 for now
        self.gamma = 1.05 * np.sqrt(self.n_algos * np.log(self.n_algos) / self.partition) 
        self.sample_parameters()


    def learning_rate(self):
        self.n_episodes = np.sum(np.array([self.algorithms[i].n_episodes for i in range(self.n_algos)]))
        return np.sqrt(np.log(self.n_algos)/(self.partition*self.n_algos))

    def sample_prob(self):
        numerator = np.exp(self.eta*(self.reward_grid))
        self.current_sample_prop = (1-self.gamma) * numerator/np.sum(numerator) + self.gamma / self.n_algos
    

    def sample_parameters(self):
        self.current_episode_reward = 0 
        self.current_T_max = self._rand_choice_nb(self.T_max_grid, self.current_sample_prop)
        self.current_T_max_index = np.where(self.current_T_max == self.T_max_grid)[0]
        self.current_algorithm = self.algorithms[self.current_T_max_index]
    
    def reset(self, s):
        self.s = s 

        for i in range(self.n_algos):
            self.algorithms[i].reset(s)
        
        self.reward_grid = np.zeros(self.n_algos,np.float64) # For sampling the algorithms
        self.reward_zero_one = np.zeros(self.n_algos,np.float64)
        self.current_sample_prop = np.ones(self.n_algos,np.float64)/self.n_algos
        self.partition = 1
        self.i = 0
        self.t = 0 
        self.hist_probs = np.zeros((10**7, self.n_algos)) # Create large array to populate historic sample proportions

        # Define Paramters as in EXP3.P (but anytime version)
        self.eta = 0.95 * np.sqrt(np.log(self.n_algos) / (self.partition * self.n_algos))
        self.beta = np.sqrt(np.log(self.n_algos/ self.delta) / (self.partition * self.n_algos)) # delta = 0.05 for now
        self.gamma = 1.05 * np.sqrt(self.n_algos * np.log(self.n_algos) / self.partition)

    def play(self, state, reward, tau):
        if self.i <= self.H:
            self.hist_probs[self.t, :] = self.current_sample_prop 
            self.i += 1
            self.current_episode_reward +=  reward/tau # Accumulated reward
            action, policy = self.current_algorithm.play(state, reward, tau) # play current algo
            self.t += tau
            return action, policy
        
        else:
            self.i = 1
            self.hist_probs[self.t, :] = self.current_sample_prop 
            self.eta = 0.95 * np.sqrt( np.log(self.n_algos) / (self.partition * self.n_algos))
            self.beta = np.sqrt(np.log(self.n_algos/ self.delta) / (self.partition * self.n_algos))
            self.gamma = 1.05 * np.sqrt(self.n_algos* np.log (self.n_algos) / self.partition)

            self.reward_zero_one = np.zeros(self.n_algos, np.float64)
            self.reward_zero_one[self.current_T_max_index] = self.current_episode_reward*1/self.H


            self.reward_grid += (self.beta)/self.current_sample_prop[self.current_T_max_index]
            self.reward_grid[self.current_T_max_index] += (self.current_episode_reward*1/self.H)/self.current_sample_prop[self.current_T_max_index]
            self.sample_prob()
            self.sample_parameters()
            self.current_algorithm.s = state # update state of current algo

            self.partition += 1
            self.current_episode_reward = 0.0

            #Recurssion note allowd in numba :// do the proper loop instead
            self.current_episode_reward +=  reward/tau # Accumulated reward
            action, policy = self.current_algorithm.play(state, reward, tau) # play current algo
            self.t += tau
            return action, policy
        
    def _rand_choice_nb(self, arr, prob):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


def ucrl_smdp_wrapper(nS, nA, delta,imprv,T_max_grid):
    ls = typed.List()
    for t in T_max_grid:
        ls.append(UCRL_SMDP(nS,nA,delta,imprv=imprv,T_max = t))
    return ls

def bus3_wrapper(nS, nA, delta,H,imprv, T_max_grid):
    return BUS3(nS, nA , H ,delta,imprv,T_max_grid,algorithms = ucrl_smdp_wrapper(nS, nA, delta, imprv,T_max_grid))



    # The next algorithm is inspired by Neu et. al. (MDP-EXP3)
spec = [
	('nS', int64),
	('T_max', int64),
	('nA', int64),
	('delta', float64),
    ('s', int64),
	('tau', float64),
    ('policy', int64[:]),
    ('i', int64), # decision steps
    ('last_action', int64),
    ('current_sample_prop', float64[:,:]),
    ('action', int64),
    ('current_reward', float64),
    ('current_tau', int64),
    ('action_grid',int64[:]),
    ('eta',float64),
    ('gamma',float64),
    ('temp',float64[:,:]),
    ('temp_prod',float64[:,:]),
    ('N', int64),
    ('mu', float64[:]),
    ('mu_st', float64[:]),
    ('history_matrix',float64[:,:,:]),
    ('history_matrix2',float64[:,:,:]),
    ('history_action',int64[:]),
    ('history_state',int64[:]),
    ('eneren',float64[:]),
    ('rhat', float64[:,:]),
    ('tauhat', float64[:,:]),
    ('qhat', float64[:,:]),
    ('qhatdiff', float64[:,:]),
    ('rhohat', float64),
    ('inv', float64[:,:,:]),
    ('P', float64[:,:,:]),
    ('current_P', float64[:,:]),
    ('identity', float64[:,:,:]),
    ('prod', float64[:,:]),
    ('w', float64[:,:]),
    ('w_ny', float64[:,:]),
    ('diff', float64[:,:]),
    ("a_hist", int64),
    ("s_hist", int64),
    ("mat_solve", float64[:,:]),
    ("zero_one", float64[:]),
    ("V", float64[:]),
    ("V_temp", float64[:])

    ]

@jitclass(spec = spec)
class SMDP_EXP3:
    def __init__(self, nS, nA ,N,P):
        self.nS = nS
        self.nA = nA
        self.current_sample_prop = np.ones((self.nA,self.nS),np.float64)/self.nA
        self.N = N
        self.i = 1 # for making formulas work.
        self.action_grid = np.arange(0,self.nA,dtype = np.int64)
        self.mu = np.ones(self.nS,np.float64)
        self.mu_st = np.ones(self.nS,np.float64)
        self.P = P
        self.current_P = np.zeros((self.nS,self.nS),np.float64)
        self.history_matrix = np.zeros((self.N-1,self.nS,self.nS),np.float64)
        self.history_action = np.zeros(self.N-1,np.int64)
        self.history_state = np.zeros(self.N-1,np.int64)
        self.a_hist = 0
        self.s_hist = 0
        self.history_matrix2 = np.zeros((self.N-1,self.nS,self.nS),np.float64)
        self.rhat = np.zeros((self.nS,self.nA),np.float64)
        self.tauhat = np.zeros((self.nS,self.nA),np.float64)
        self.qhat  = np.zeros((self.nS,self.nA),np.float64)
        self.qhatdiff  = np.zeros((self.nS,self.nA),np.float64)
        self.V = np.zeros(self.nS,np.float64)
        self.V_temp = np.zeros(self.nS,np.float64)

        self.diff  = np.zeros((self.nS,self.nA),np.float64)
        self.identity = np.zeros((self.nS,self.nA,self.nS), np.float64)
        self.prod = np.zeros((self.nS,self.nS), np.float64)
        self.inv = np.zeros((self.nS,self.nA,self.nS), np.float64)

        # Define Paramters as in EXP3.P (but anytime version)
        self.eta = 0.95 * np.sqrt(np.log(self.nA) / (self.i * self.nA))
        self.gamma = 1.05 * np.sqrt(self.nA * np.log(self.nA) / self.i) 
        self.w = np.ones((self.nA,self.nS),np.float64)
        self.w_ny = np.ones((self.nA,self.nS),np.float64)

        self.eneren = np.zeros(self.nS,np.float64)
        self.mat_solve = np.ones((self.nS+1,self.nS),np.float64)
        self.zero_one = np.zeros(self.nS+1, np.float64)
        self.zero_one[self.nS] = 1.0
        self.i = 0 # For zero indexing.

    


    def sample_prob(self):
        for a in range(self.nA):
            for s in range(self.nS):
                self.current_sample_prop[a,s] = (1-self.gamma) * self.w[a,s]/np.sum(self.w[:,s]) + self.gamma / self.nA
    

    def sample_parameters(self):
        self.action = self._rand_choice_nb(self.action_grid, self.current_sample_prop[:,self.s])
        return self.action

    def reset(self, s):
        self.s = s # set initial state to first state. 
        self.current_sample_prop = np.ones((self.nA,self.nS),np.float64)/self.nA
        self.i = 1 # for making formulas work.
        self.action_grid = np.arange(0,self.nA,dtype = np.int64)
        self.mu = np.ones(self.nS,np.float64)
        self.mu_st = np.ones(self.nS,np.float64)
        self.current_P = np.zeros((self.nS,self.nS),np.float64)

        self.history_matrix = np.zeros((self.N-1,self.nS,self.nS),np.float64)
        self.history_action = np.zeros(self.N-1,np.int64)
        self.history_state = np.zeros(self.N-1,np.int64)
        self.a_hist = 0 # for transitioning
        self.s_hist = s # for transitioning
        self.history_matrix2 = np.zeros((self.N-1,self.nS,self.nS),np.float64)
        self.rhat = np.zeros((self.nS,self.nA),np.float64)
        self.tauhat = np.zeros((self.nS,self.nA),np.float64)
        self.qhat  = np.zeros((self.nS,self.nA),np.float64)
        self.qhatdiff  = np.zeros((self.nS,self.nA),np.float64)
        self.V = np.zeros(self.nS,np.float64)
        self.V_temp = np.zeros(self.nS,np.float64)

        self.diff  = np.zeros((self.nS,self.nA),np.float64)
        self.identity = np.zeros((self.nS,self.nA,self.nS), np.float64)
        self.prod = np.zeros((self.nS,self.nS), np.float64)
        self.inv = np.zeros((self.nS,self.nA,self.nS), np.float64)

        # Define Paramters as in EXP3.P (but anytime version)
        self.eta = 0.95 * np.sqrt(np.log(self.nA) / (self.i * self.nA))
        self.gamma = 1.05 * np.sqrt(self.nA * np.log(self.nA) / self.i) 
        self.w = np.ones((self.nA,self.nS),np.float64)
        self.w_ny = np.ones((self.nA,self.nS),np.float64)

        self.eneren = np.zeros(self.nS,np.float64)
        self.mat_solve = np.ones((self.nS+1,self.nS),np.float64)
        self.zero_one = np.zeros(self.nS+1, np.float64)
        self.zero_one[self.nS] = 1.0
        self.i = 0 # For zero indexing.


    def play(self, state, reward, tau):

        # 1) Playing for N decision steps.         
        action = self.sample_parameters() # Sample action. 
        self.s = state # update state of current algo
        self.current_reward = reward
        self.current_tau = tau
        
        self.last_action = action
        # Generate history up till now. 
        temp = np.zeros((self.nS,self.nS),np.float64)
        for s in range(self.nS):
            for sp in range(self.nS):
                temp[s,sp]  = self.current_sample_prop[:,s] @ self.P[s,:,sp]
        if self.i < self.N - 1 :
            self.history_matrix2[self.i,:,:] = temp  # set as the same. 
            self.history_action[self.i] = action
            self.history_state[self.i] = self.s

        # 4) if i >= N.
        if self.i >= self.N - 1:
        # 5) Compute mu
            # Update current t-N history
            self.a_hist = self.history_action[0] # first in line
            self.s_hist = self.history_state[0] 
            # History t-N up till t-1
            for i in range(0,self.N-2):   
                self.history_matrix[i,:,:] = self.history_matrix2[i+1,:,:]

            # Compute history of P^pi of current t
            self.history_matrix[self.N-2,:,:] = temp 
            
            # Create matrix product of P^pi's
            self.prod =  np.identity(self.nS,np.float64)
            for i in range(self.N-2,0,-1): # do not include newest.
                temp_prod = self.history_matrix2[i,:,:] @ self.prod
                self.prod = temp_prod 
            # Compute mu
            self.eneren = np.zeros(self.nS,np.float64)
            self.eneren[self.s_hist] = 1.0
            self.mu = self.eneren @ self.P[:,self.a_hist,:] @ self.prod
            # Update history matrix
            self.history_matrix2 = self.history_matrix # update for next run.
            # update history of actions and state
            for i in range(0,self.N-2):   
                self.history_action[i] = self.history_action[i+1]
                self.history_state[i] = self.history_state[i+1]
            self.history_action[self.N-2] = action
            self.history_state[self.N-2] = self.s

            # 6) Compute r hat and tau hat. Compute q hat. 
            self.rhat = np.zeros((self.nS,self.nA),np.float64)
            self.tauhat = np.zeros((self.nS,self.nA),np.float64)
            # Update weighted estimates.
            # Firstly a sanity check
            self.rhat[self.s,action] = reward /(self.current_sample_prop[action,self.s]*self.mu[self.s]) #use tau to avoid overflow
            self.tauhat[self.s,action] = tau /(self.current_sample_prop[action,self.s]*self.mu[self.s])
            # Compute current P^pi
            self.current_P = np.zeros((self.nS,self.nS),np.float64)
            for s in range(self.nS):
                for sp in range(self.nS):
                    self.current_P[s,sp]  = np.sum(self.current_sample_prop[:,s] * self.P[s,:,sp])
            # Calculate stationary distribution heuristically via matrix power (assume consergence).
            self.mu_st = self.eneren @ np.linalg.matrix_power(self.current_P, 50) 
            # By least squares solver
            #self.mat_solve[:self.nS,:] = np.identity(self.nS,np.float64)-self.current_P            
            #self.mu_st = np.linalg.lstsq(self.mat_solve,self.zero_one,rcond = 10**(-120))[0]
            
            # Find rho by importance sampled estimates. 
            self.rhohat = 0
            for s in range(self.nS):
                for a in range(self.nA):
                    self.rhohat += self.mu_st[s] * self.current_sample_prop[a,s] * self.rhat[s,a]
            # Calculate difference
            self.diff = np.zeros((self.nS,self.nA),np.float64)
            self.diff[self.s,action] = self.rhat[self.s,action] - self.rhohat * self.tauhat[self.s,action] 


            # Find value function: 
            differ = np.identity(self.nS) - self.current_P
            # compute temporary v
            for s in range(self.nS):
                self.V_temp[s] = np.sum(self.rhat[s,:] * self.current_sample_prop[:,s] - self.rhohat * self.tauhat[s,:] * self.current_sample_prop[:,s])
            #Compute actual V
            self.V = np.linalg.lstsq(differ, self.V_temp,rcond = 10**(-120))[0]
            print(self.V)
            # Find Qhat according to (5)
            for s in range(self.nS):
                for a in range(self.nA):
                    self.qhat[s,a] += self.rhat[s,a] - self.rhohat*self.tauhat[s,a] + self.P[s,a,:] @ self.V # increment
            # The last step increments qhat. We then calculate relative diff (see footnote 6):
            # i.e. For nummerical stable.
            for s in range(self.nS):    
                self.qhatdiff[s,:] = self.qhat[s,:] - np.max(self.qhat[s,:])*np.ones(self.nA,np.float64)
            
            # 7) Update w.
            for a in range(self.nA):
                self.w_ny[a,self.s] = (self.w[a,self.s]+10**(-10)) * np.exp(self.eta*( self.qhatdiff[self.s,a]))
            self.w = self.w_ny  # small number added to not zero next time above. +10**(-10) 
        # Updates.
        self.i = self.i + 1
        self.eta = 0.95 * np.sqrt(np.log(self.nA) / (self.i * self.nA))
        self.gamma = 1.05 * np.sqrt(self.nA * np.log(self.nA) / self.i) 


        policy = np.array(self.nS, np.int64) 
        self.sample_prob()

        return action, policy
        
    def _rand_choice_nb(self, arr, prob):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
