# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2023.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np
import copy as cp


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	UCRL2

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################


# A simple implementation of the UCRL2 algorithm from Jacksh et al. 2010 with improved L1-Laplace confidence intervals.
class UCRL2_L:
	def __init__(self, nS, nA, delta = 0.05, T_max = None):
		self.nS = nS
		self.nA = nA
		self.delta = delta / (2* nS * nA)# As used in proof of lemma 5 in the original paper.
		self.s = None
		self.n_episodes = 0 # added
		self.t = 1  #added counter 
		self.current_bias_estimate = np.zeros(self.nS) # for speed-up of EVI

		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
		self.vk = np.zeros((self.nS, self.nA)) # Number of occurences of (s, a) in the current episode.

		# The "estimates" variables:
		self.hatP = np.zeros((self.nS, self.nA, self.nS)) # Estimate of the transition matrix.
		self.hatR = np.zeros((self.nS, self.nA))
		
		# Confidence intervals:
		self.confR = np.zeros((self.nS, self.nA))
		self.confP = np.zeros((self.nS, self.nA))

		# The current policy (updated at each episode).
		self.policy = np.zeros((self.nS,), dtype=int)
		


	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]

	# Update the confidence intervals. Set with Laplace-L1 confidence intervals!
	def confidence(self):
		d = self.delta
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.confR[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.confP[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	# From UCRL2 jacksh et al. 2010.
	def max_proba(self, sorted_indices, s, a):
		
		min1 = min([1, self.hatP[s, a, sorted_indices[-1]] + (self.confP[s, a] / 2)])
		max_p = np.zeros(self.nS)
			
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(self.hatP[s, a])
			max_p[sorted_indices[-1]] += self.confP[s, a] / 2
			l = 0 
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
				
		return max_p


	# The Extended Value Iteration, perform an optimisitc VI over a set of MDP.
	#Note, changed fixed epsilon to 1/sqrt(t)
	def EVI(self, max_iter = 2*10**3):
		niter = 0
		epsilon = 1/np.sqrt(self.t)
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]

		# The variable containing the optimistic policy estimate at the current iteration.
		policy = np.zeros(self.nS, dtype=int)

		# Initialise the value and epsilon as proposed in the course.
		V0 = self.current_bias_estimate # NB: setting it to the bias obtained at the last episode can help speeding up the convergence significantly!, Done!
		V1 = np.zeros(self.nS)
		sorted_indices = np.argsort(V0)

		# The main loop of the Value Iteration algorithm.
		while True:
			niter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					maxp = self.max_proba(sorted_indices, s, a)
					temp = min(1, self.hatR[s, a] + self.confR[s, a]) + sum([V * p for (V, p) in zip(V0, maxp)])
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
			if niter > max_iter:
				print("No convergence in EVI after: ", max_iter, " steps!", maxp)
				return policy

	# To start a new episode (init var, computes estmates and run EVI).
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

	# To reinitialize the model and a give the new initial state init.
	def reset(self, init=0):
		# The "counter" variables:
		self.Nk = np.zeros((self.nS, self.nA), dtype=int) # Number of occurences of (s, a) at the end of the last episode.
		self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=int) # Number of occureces of (s, a, s').
		self.Rsa = np.zeros((self.nS, self.nA)) # Cumulated reward observed for (s, a).
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
		self.t = 1

		
		# Start the first episode.
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state, reward,tau):
		if self.last_action >= 0: # Update if not first action.
			self.Nsas[self.s, self.last_action, state] += 1
			self.Rsa[self.s, self.last_action] += reward
		
		action = self.policy[state]
		if self.vk[state, action] > max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		
		# Update the variables:
		self.vk[state, action] += 1
		self.s = state
		self.last_action = action
		self.t += 1

		return action, self.policy



class UCRL2(UCRL2_L):
	def confidence(self):
		d = self.delta
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.confR[s, a] = np.sqrt((3.5/n) * np.log(2 * self.nS * self.nA * self.t / d))
				self.confP[s, a] = np.sqrt((14 * self.nS/n * np.log(2*self.nA*self.t/d)))