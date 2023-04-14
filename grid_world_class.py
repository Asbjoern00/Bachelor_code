# Packages:
import numpy as np

##########################################################################################
# Define options to be a function of grid, coordinates and steps
# Takes the agent a length of m steps in one direction.
def options(map,x,y,step):
	up = map[x-step, y]
	right = map[x, y+step]
	down = map[x+step, y]
	left = map[x, y-step]
	return up,right,down,left

# To do: Give a parameter T_max in order to get options in.  
# Actions: '0' = 'go up', '1' = 'go right', '2'  = 'go down' thus '3' = 'go left'
# State 0 is upper left, state nS - 1 is lower right.
# The agent is teleported back to anay other state except for the target state 'nS - 1'
class grid_world():

	def __init__(self,nS,T_max):
		# Construction of the matrix :
		self.nS = nS
		d_sqrt = int(nS**(1/2)) # square root of number of states 
		nS = self.nS
		self.nA = 4 # 4 options exists. 
		self.map = np.full((d_sqrt+2,d_sqrt+2),-1) # plus to as we add on both sides
		center = np.array(range(0,self.nS)).reshape(d_sqrt,d_sqrt)
		self.map[1:d_sqrt+1,1:d_sqrt+1] = center
		map = self.map
		# Grid is now created.

		# We build the transitions matrix P (note that this shouls hold for options).
		self.P = np.zeros((nS, 4, nS))
		self.tau = np.zeros((nS, 4, nS))
		self.tau_bar = np.zeros((nS,4)) # expected holding time.
		self.P_eq = np.zeros((nS,4,nS))

		for s in range(nS):
			# We need a notion of states between e.g. up and current state.
			# Find number of time steps
			row_index=int(np.where(map==s)[0])
			col_index=int(np.where(map==s)[1])
			# states under option.
			down = map[row_index+1:-1,col_index]
			right = map[row_index,col_index+1:-1]
			up = np.flip(map[1:row_index,col_index]) # necessary to flip this (later index)
			left = np.flip(map[row_index,1:col_index]) # necessary to flip this
			# length to wall (subtract one to get actual number of states).
			m_down = len(map[row_index+1:,col_index])
			m_right = len(map[row_index,col_index+1:])
			m_up = len(map[:row_index,col_index])
			m_left = len(map[row_index,:col_index])



			# option setting.
			# option 0: up.
			o = 0
			if m_up>T_max: # further away than maximum.
				self.P[s, o, up[:T_max]] = 1/(T_max) # use T_max in this case - ommit -1 - T_max elements.
				self.tau[s,o,up[:T_max]] = range(1,T_max+1) # tau
			if m_up<=T_max and m_up>1: # closer than max number of steps.
				self.P[s, o, up] = 1/(m_up-1) # as there is one state less avaivable - also ommit -1.  
				self.tau[s,o,up] = range(1,m_up) # tau
			# find expected holding time for state option pair (see appendix F).
			self.tau_bar[s,o]=(T_max + 1)/2
			# define equivalent probabilities
			self.P_eq[s,o] = (1-0.1)/(self.tau_bar[s,o])*self.P[s,o] 
			if m_up == 1: # length one to wall (i.e. -1)
				self.P[s, o, s] = 1.0 # Still certain transition.
				self.tau[s,o,s] = 1.0
			self.P_eq[s,o,s] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,s]-1)+1
 
					
			# option 1: right.
			o = 1
			if m_right>T_max: # further away than maximum.
				self.P[s, o, right[:T_max]] = 1/(T_max) # use T_max in this case - ommit -1 - T_max elements.
				self.tau[s,o,right[:T_max]] = range(1,T_max+1) # tau
			if m_right<=T_max and m_right>1: # closer than max number of steps.
				self.P[s, o, right] = 1/(m_right-1) # as there is one state less avaivable - also ommit -1.  
				self.tau[s,o,right] = range(1,m_right) # tau
			# find expected holding time for state option pair (see appendix F).
			self.tau_bar[s,o]=(T_max + 1)/2
			# define equivalent probabilities
			self.P_eq[s,o] = (1-0.1)/(self.tau_bar[s,o])*self.P[s,o] 
			if m_right == 1: # length one to wall (i.e. -1)
				self.P[s, o, s] = 1.0 # Still certain transition.
				self.tau[s,o,s] = 1.0
			self.P_eq[s,o,s] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,s]-1)+1


			# option 2: down.		
			o = 2
			if m_down>T_max: # further away than maximum.
				self.P[s, o, down[:T_max]] = 1/(T_max) # use T_max in this case - ommit -1 - T_max elements.
				self.tau[s,o,down[:T_max]] = range(1,T_max+1) # tau

			if m_down<=T_max and m_down>1: # closer than max number of steps.
				self.P[s, o, down] = 1/(m_down-1) # as there is one state less avaivable - also ommit -1.  
				self.tau[s,o,down] = range(1,m_down) # tau
			# find expected holding time for state option pair (see appendix F).
			self.tau_bar[s,o]=(T_max + 1)/2
			# define equivalent probabilities
			self.P_eq[s,o] = (1-0.1)/(self.tau_bar[s,o])*self.P[s,o] 
			if m_down == 1: # length one to wall (i.e. -1)
				self.P[s, o, s] = 1.0 # Still certain transition.
				self.tau[s,o,s] = 1.0
			self.P_eq[s,o,s] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,s]-1)+1


			# option 3: left.	
			o = 3
			if m_left>T_max: # further away than maximum.
				self.P[s, o, left[:T_max]] = 1/(T_max) # use T_max in this case - ommit -1 - T_max elements.
				self.tau[s,o,left[:T_max]] = range(1,T_max+1) # tau

			if m_left<=T_max and m_left>1: # closer than max number of steps.
				self.P[s, o, left] = 1/(m_left-1) # as there is one state less avaivable - also ommit -1.  
				self.tau[s,o,left] = range(1,m_left) # tau
			# find expected holding time for state option pair (see appendix F).
			self.tau_bar[s,o]=(T_max + 1)/2
			# define equivalent probabilities
			self.P_eq[s,o] = (1-0.1)/(self.tau_bar[s,o])*self.P[s,o] 			
			if m_left == 1: # length one to wall (i.e. -1)
				self.P[s, o, s] = 1.0 # Still certain transition.					
				self.tau[s,o,s] = 1.0
			self.P_eq[s,o,s] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,s]-1)+1

				

			# Set to teleport uniformly in the actions setting
			if s == self.nS - 1:
				for o in range(4):
					for ss in range(self.nS):
						self.P[s, o, ss] = 1/(self.nS-1) # uniformly for all except of yellow state i.e. 1/(S-1)=1/19 
						self.tau[s, o, ss] = 1.0 # always one unit of time here.
						self.tau_bar[s,o] = 1 # always one unit (teleportation)
						self.P_eq[s,o,ss] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,ss])

						if ss == self.nS-1: #if we get to the same 'yellow' state - happen with zero prob.:
							self.P[s, o, ss] = 0
							self.P_eq[s,o,ss] = (1-0.1)/(self.tau_bar[s,o])*(self.P[s,o,ss]-1)+1

			
		# We build the reward matrix R. This is supposed to only give a reward in the nS-1 state.
		self.R = np.zeros((nS, 4))
		for o in range(4):
			self.R[nS - 1, o] = 1

		# We (arbitrarily) set the initial state in the top-left corner.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, option):
		new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, option])
		tau = self.tau[self.s,option,new_s]
		reward = self.R[self.s, option]
		self.s = new_s
		return new_s, reward, tau



# Value iteration also works for equivalent MDP - this is how optimal gain is found for options.

def VI(env, max_iter = 10**3, epsilon = 10**(-4)):

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
				temp = env.R[s, a] + sum([V * p for (V, p) in zip(V0, env.P_eq[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy[s] = a
		
		# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
		gain = 0.5*(max(V1 - V0) + min(V1 - V0))
		diff  = [abs(x - y) for (x, y) in zip(V1, V0)]
		if (max(diff) - min(diff)) < epsilon:
			return niter, V0, policy, gain
		else:
			V0 = V1
			V1 = np.zeros(env.nS)
		if niter > max_iter:
			print("No convergence in VI after: ", max_iter, " steps!")
			return niter, V0, policy, gain
print(10+1)