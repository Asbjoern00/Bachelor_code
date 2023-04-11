# Packages:
import numpy as np

##########################################################################################
# To do: Give a parameter T_max in order to get options in. 
# Actions: '0' = 'go up', '1' = 'go right', '2'  = 'go down' thus '3' = 'go left'
# State 0 is upper left, state nS - 1 is lower right.
# The agent is teleported back to anay other state except for the target state 'nS - 1'
class grid_world():

	def __init__(self,nS,prim):
		self.nS = nS
		d_sqrt = int(nS**(1/2)) # square root of number of states 
		nS = self.nS
		self.nA = 4 # there are 4 actions

		self.map = np.full((d_sqrt+2,d_sqrt+2),-1) # plus to as we add on both sides
		center = np.array(range(0,self.nS)).reshape(d_sqrt,d_sqrt)
		self.map[1:d_sqrt+1,1:d_sqrt+1] = center
		map = self.map

		# We build the transitions matrix P using the map.
		self.P = np.zeros((nS, 4, nS))
		for s in range(nS):
			temp = np.where(s == map)

			x, y = temp[0][0], temp[1][0]
			up = map[x-1, y]
			right = map[x, y+1]
			down = map[x+1, y]
			left = map[x, y-1]

			# Action 0: go up.
			a = 0
			# Up
			if up == -1:
				self.P[s, a, s] += 1.0
			else:
				self.P[s, a, up] += 1.0
					
			# Action 1: go right.
			a = 1
			# Right
			if right == -1:
				self.P[s, a, s] += 1.0
			else:
				self.P[s, a, right] += 1.0
						
			# Action 2: go down.
			a = 2
			# Down
			if down == -1:
				self.P[s, a, s] += 1.0
			else:
				self.P[s, a, down] += 1.0

			# Action 3: go left.
			a = 3
			# Left
			if left == -1:
				self.P[s, a, s] += 1.0
			else:
				self.P[s, a, left] += 1.0							
				
							
			# Set to teleport uniformly
			if s == self.nS - 1:
				for a in range(4):
					for ss in range(self.nS):
						self.P[s, a, ss] = 1/(self.nS-1) # uniformly for all except of yellow state i.e. 1/(S-1)=1/19 
						if ss == self.nS-1: #if we get to the same 'yellow' state - happen with zero prob.:
							self.P[s, a, ss] = 0
			
		# We build the reward matrix R.
		self.R = np.zeros((nS, 4))
		for a in range(4):
			self.R[nS - 1, a] = 1

		# We (arbitrarily) set the initial state in the top-left corner.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, action,prim):
		if prim==True: # Introduce the actions setting.
			new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
			reward = self.R[self.s, action]
			self.s = new_s

		return new_s, reward