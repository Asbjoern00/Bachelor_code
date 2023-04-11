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

        # The "counter" variables:
        self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.
        self.i = 1

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
    
    def update_n(self):
       self.Nk += self.vk 

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
                for next_s in range(self.nS):
                    self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

        # Update the confidence intervals and policy.
        self.confidence()
        self.policy = self.EVI()



    def confidence(self):
                
        beta_tau = 
        beta_p = 
        beta_r = 