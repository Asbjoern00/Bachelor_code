# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2023.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np
import copy as cp

from numba import int64, float64,boolean
from numba.experimental import jitclass
spec = [
	('nS', int64),
	('T_max', int64),
	('nA', int64),
	('delta', float64),
    ('b_r', float64),
    ('r_max', float64),
    ('tau_min', float64),
    ('imprv', int64),
    ('sigma_r', float64),
    ('T_max', float64),
    ('episode_ended',boolean),
    ('current_bias_estimate', float64[:]),
    ('n_episodes',int64),
    ('s', int64),
    ('Nk',int64[:,:]),
    ('Nsas', int64[:,:,:]),
    ('Rsa',float64[:,:]),
    ('vk', int64[:,:]),
    ('hatR', float64[:,:]),
	('hatP', float64[:,:,:]),
    ('confR', float64[:,:]),
	('confP', float64[:,:]),
    ('policy', int64[:]),
    ('t', int64),
    ('last_action', int64)
    ]

@jitclass(spec=spec)
class UCRL2:
	def __init__(self, nS, nA, delta = 0.05, T_max = 1, imprv = 0):
		self.nS = nS
		self.nA = nA
		self.delta = delta / (2* nS * nA)# As used in proof of lemma 5 in the original paper. 
		self.n_episodes = 0 # added
		self.t = 1  #added counter 
		self.current_bias_estimate = np.zeros(self.nS, dtype=np.float64) # for speed-up of EVI
		self.T_max = T_max
		self.imprv = imprv

		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA), dtype=np.float64)
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA), dtype=np.float64)
		self.confP = np.zeros((self.nS, self.nA), dtype=np.float64)

		# The current policy (updated at each episode).
		self.policy = np.zeros(self.nS, dtype=np.int64)
		


	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in np.arange(self.nS):
			for a in np.arange(self.nA):
				self.Nk[s, a] += self.vk[s, a]

	# Update the confidence intervals. Set with Laplace-L1 confidence intervals!
	def confidence(self):
		if self.imprv == 1:
			d = self.delta
			for s in np.arange(self.nS):
				for a in np.arange(self.nA):
					n = np.max(np.array([1, self.Nk[s, a]]))
					self.confR[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					self.confP[s,a] = np.sqrt( (2*(1+1/n) * (self.nS*np.log(2) + np.log(np.sqrt(n+1)*self.nS*self.nA/d) ) / (n) ))
		if self.imprv == 0:
			d = 4 * self.delta #times 4 for making comparible with other algs
			for s in np.arange(self.nS):
				for a in np.arange(self.nA):
					n = max(1, self.Nk[s, a])
					self.confR[s, a] = np.sqrt((3.5/n) * np.log(2 * self.t / d))
					self.confP[s, a] = np.sqrt((14 * self.nS/n * np.log(2*self.nA*self.t/d)))
		
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# From UCRL2 jacksh et al. 2010.
	def max_proba(self, sorted_indices, s, a):
		
		min1 = np.min(np.array([1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2)]))
		max_p = np.zeros(self.nS, np.float64)
			
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
		epsilon = 1/np.sqrt(self.t)
		#epsilon = 0.01
		sorted_indices = np.arange(self.nS, dtype=np.int64)

		# Initialise the value and epsilon as proposed in the course.
		V0 = self.current_bias_estimate # NB: setting it to the bias obtained at the last episode can help speeding up the convergence significantly!, Done!
		V1 = np.zeros(self.nS, np.float64)
		r_tilde = self.confR + self.hatR
		# The main loop of the Value Iteration algorithm.
		while True:
			niter += 1
			V0 = np.ascontiguousarray(V0)
			mat_products = np.zeros((self.nS, self.nA))
			
			for s in range(self.nS):
				for a in range(self.nA):
					maxp = self.max_proba(sorted_indices, s, a).T
					mat_products[s,a] += maxp@V0
				
				V1[s] += np.max(r_tilde[s,:]+((mat_products[s,:])))
			
			diff  = np.abs(V1-V0)
			if (np.max(diff) - np.min(diff)) < epsilon:
				policy = np.zeros(self.nS, dtype=np.int64)
				self.current_bias_estimate = V1
				for s in range(self.nS):
					candidate_actions = [] # Randomize choices in case of multiple optimal
					for a in range(self.nA):
						if np.isclose(V1[s] ,r_tilde[s,a]+mat_products[s,a]):
							#policy[s] = a
							candidate_actions.append(a)
						if a == self.nA -1:
							policy[s] = np.random.choice(np.array(candidate_actions, dtype=np.int64))
							self.current_bias_estimate = V1 - np.mean(V1)
				return policy
			
			else:
				V0 = V1
				V1 = np.zeros(self.nS, dtype=np.float64)
				sorted_indices = np.argsort(V0)

			if niter > max_iter:
				print("No convergence in EVI after:" ,max_iter,  "steps!. Actual diff was", max(diff)-min(diff), "and epsilon =", epsilon, 
                      "current bias estimate =" , self.current_bias_estimate)
				for s in range(self.nS):
						policy[s] = np.argmax(r_tilde[s,:]+((mat_products[s,:])))
						self.current_bias_estimate = V1 - np.mean(V1)
				return policy

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # We update the counter Nk.
		self.vk = np.zeros((self.nS, self.nA), dtype=np.int64)
		self.n_episodes +=1 # add to episode counter

		# Update estimates, note that the estimates are 0 at first, the optimistic strategy making that irrelevant.
		for s in np.arange(self.nS):
			for a in np.arange(self.nA):
				div = np.max(np.array([1, self.Nk[s, a]]))
				self.hatR[s, a] = self.Rsa[s, a] / div
				for next_s in np.arange(self.nS):
					self.hatP[s, a, next_s] = self.Nsas[s, a, next_s] / div

		# Update the confidence intervals and policy.
		self.confidence()
		self.policy = self.EVI()
		#print(f"New episode started at {self.t}")

	# To reinitialize the model and a give the new initial state init.
	def reset(self, init=0):
		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA), dtype=np.float64) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA), dtype=np.int64) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA), dtype=np.float64)
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA), dtype=np.float64)
		self.confP = np.zeros((self.nS, self.nA), dtype=np.float64)

		# The current policy (updated at each episode).
		self.policy = np.zeros(self.nS, dtype=np.int64)

		# Set the initial state and last action:
		self.s = init
		self.last_action = -1
		self.t = 1

		
		# Start the first episode.
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state, reward,tau):
		if self.last_action >= 0: # Update if not first action.
			self.Nsas[self.s, self.last_action, state] += 1
			self.Rsa[self.s, self.last_action] += reward
		
		action = self.policy[state]
		if self.vk[state, action] > np.max(np.array([1, self.Nk[state, action]])): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		
		# Update the variables:
		self.vk[state, action] += 1
		self.s = state
		self.last_action = action
		self.t += 1

		return action, self.policy
