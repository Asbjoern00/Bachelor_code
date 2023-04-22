import numpy as np
# PURPOSE: IMPLEMENTATION OF EPSILON-GREEDY ALGORITHM. 
class GREEDY():
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        # The "counter" variables:
        self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
        self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
        
        #Counter variables that might be unnecessary
        self.i = 1

        # The "estimates" variables:
        self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
        self.hatR = np.zeros((self.nS, self.nA)) # Estimate of rewards
        
        # The current policy (updated at each decision step).
        self.policy = np.zeros((self.nS,), dtype=int)

    
    def epsilon(self):
        return 1/np.sqrt(self.i) # return 1/sqrt(decision step)
    
    def updates(self):
        # Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                self.hatR[s, a] = self.Rsa[s, a] / div
                #self.hattau[s,a] = self.tausa[s,a] / div
                for next_s in range(self.nS):
                    self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

    def play(self):
        a = 
        