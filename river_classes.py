import numpy as np
        
# index indicates where the node is in the river. index = 0 is equivalent to left bank and index = L is equivalent to right bank
# rewards and transition_probs should be dicts of the form
# rewards = {"L" : r_L, "R" : r_R} or L replaced w. R for indicating going right
# transitions_probs = {"L": np.array([p^L_{index-1}, p^L_{index}, p^L_{index+1}],
#                      "R": np.array([p^R_{index-1}, p^R_{index}, p^R_{index+1}]}
# Where p^A_{i} gives the transition probability from the current state to state i under action A. We can input only 3 values instead of transition matrix
# As all other probabilities in River-Swim are zer
class RiverNode:
    def __init__(self, rewards, transition_probs):
        #Set rewards and transition probabilities for a single node
        self.rewards = rewards
        self.transition_probs = transition_probs
        self.times_visited = 0
        self.actions_taken = {"L": 0 ,"R": 0}
        self.emperical_reward = {"L":0, "R":0}
        self.emperical_transitions = {"L":{"-1":0, "0":0, "1":0}, "R":{"-1":0, "0":0, "1":0}}
        self.times_since_last_play = {"L":0, "R":0}

    # Visits node and returns transition as well as reward
    def visit(self, action):
        # Update parameters of node 
        self.times_visited +=  1
        self.actions_taken[action] += 1
        self.emperical_reward[action] += self.rewards[action]
        # Sample next state and update transition 
        sample = np.random.choice([-1,0,1], p = self.transition_probs[action])
        self.emperical_transitions[action][str(sample)] += 1
        for play in ["R","L"]:
            if action != play:
                self.times_since_last_play[play] +=1
            else:
                self.times_since_last_play[play] = 0

        return sample,self.rewards[action]

    def get_estimate(self):
        reward_estimate = {}
        transition_estimate = {}
        for action in ["L","R"]:
            reward_estimate[action] = self.emperical_reward[action]/max(1,self.actions_taken[action])
            transition_estimate[action] = np.array([self.emperical_transitions[action][next_s]/max(self.actions_taken[action],1) for next_s in ["-1","0","1"]])
        return RiverNode(rewards=reward_estimate, transition_probs=transition_estimate)
    def reset_node(self):
        #Set rewards and transition probabilities for a single node
        self.times_visited = 0
        self.actions_taken = {"L": 0 ,"R": 0}
        self.emperical_reward = {"L":0, "R":0}
        self.emperical_transitions = {"L":{"-1":0, "0":0, "1":0}, "R":{"-1":0, "0":0, "1":0}}



class RiverSwimEnvironment:
    def __init__(self, gamma, nodes = None, n_states = None, reward_leftbank ={"L":0.05, "R":0}, reward_other = {"L":0, "R":0}, reward_rightbank = {"L":0, "R":1},
    trans_leftbank = {"L":np.array([0,1,0]),"R":np.array([0,0.6,0.4])}, trans_other = {"L":np.array([1,0,0]),"R":np.array([0.05,0.55,0.4])},
    trans_rightbank = {"L":np.array([1,0,0]),"R":np.array([0.4,0.6,0])}):
        self.nodes = nodes
        if nodes is None:
            left_node = RiverNode(rewards=reward_leftbank, transition_probs=trans_leftbank) # Generate left bank node
            right_node = RiverNode(rewards=reward_rightbank, transition_probs=trans_rightbank) # generate right bank node
            self.nodes = [left_node] + ([RiverNode(rewards=reward_other, transition_probs=trans_other) for i in range(n_states-2)]) + [right_node] # Make a list of nodes
        self.n_states = len(self.nodes)
        self.gamma = gamma # set discount
        self.current_state = 0 # Set initial state equal to 0 
    
    def reset_all(self):
        for node in self.nodes:
            node.reset_node()
            self.current_state = 0 

    #Method for constructing reward-vector, transition-matrix and Value-vector given a policy.
    
    # Policy should be of the form np.array(["L","R","L"...]) such that policy[i] = action taken in state i
    # Mostly legacy for constructing all_L and all_R

    def gen_matrices(self, policy):
        reward_vector = np.array([self.nodes[i].rewards[policy[i]] for i in range(self.n_states)]) # Generates 

        transitionmatrix = np.zeros((self.n_states+1, self.n_states+1)) # Initialise 0-matrix for transition probabilities. Pad edges with +1
        
        for i in range(self.n_states):
            transitionmatrix[i, [i-1,i,i+1]] = self.nodes[i].transition_probs[policy[i]] # Fill out transition_matrix
        
        transitionmatrix = np.array(transitionmatrix)[0:self.n_states,0:self.n_states] # Make a numpy-array and throw padding away
        
        value_vector = np.linalg.inv(np.identity(self.n_states)-self.gamma*transitionmatrix) @ reward_vector # Calculate value of policy

        return {"reward":reward_vector, "transition":transitionmatrix, "value":value_vector} # Output dict containing reward vektor, transition matrix and value vector
    
    # Generates transition/reward of all_L and all_R policies. All other (stationary) policies can be derived from taking a combination of these
    def gen_all(self):
        self.all_R = self.gen_matrices(np.repeat("R", self.n_states))
        self.all_L = self.gen_matrices(np.repeat("L", self.n_states))

    #Does the same as gen_matrices, but for stochastic policies. Should be called after gen_all()
    def eval_stochastic_policy(self, p_go_right=np.array([0.5,0.5,0.5,0.5])):
        reward_vector = self.all_R["reward"] * p_go_right + self.all_L["reward"]*(1-p_go_right)
        transition_matrix = self.all_R["transition"]*p_go_right + self.all_L["transition"]*(1-p_go_right)
        value_vector = np.linalg.inv(np.identity(self.n_states)-self.gamma*transition_matrix) @ reward_vector # Calculate value of policy
        return {"reward":reward_vector, "transition":transition_matrix, "value":value_vector} # Output dict containing reward vektor, transition matrix and value vector

    
    def value_iteration(self,epsilon):
        n = 0 # set n = 0
        rmax = max([max(self.nodes[i].rewards.values()) for i in range(self.n_states)]) # Get rmax
        V_n = np.zeros(self.n_states) # Set V_n to 0 (arbitrary)
        V_np1  = np.ones(self.n_states)*rmax*(1/(1-self.gamma)) # Set V_1 = rmax/(1-gamma)*(1-vector)
        
        #Generate reward/transition-matrices
        all_R = self.gen_matrices(np.repeat("R", self.n_states))
        all_L = self.gen_matrices(np.repeat("L", self.n_states))

        while(np.max(np.abs(V_np1-V_n))>= (1-self.gamma)/(2*self.gamma)*epsilon):
            V_n = V_np1
            #Compute V_{n+1}(s) = max (r(s,a) + gamma*sum(p(x|s,a)*V_{n}(x))) for each s, by doing creating transition/reward-matrices for "L", "R"-actions in each state.
            V_np1_R = all_R["reward"] + self.gamma*all_R["transition"]@V_np1
            V_np1_L = all_L["reward"] + self.gamma*all_L["transition"]@V_np1
            V_np1 = np.maximum(V_np1_R,V_np1_L)
            n=n+1
        final_policy = np.where(V_np1_R == V_np1, "R","L") # Get final policy
        self.n_iter_VI = n
        self.final_policy_VI = final_policy
        self.final_valuef_VI = V_np1

    def policy_iteration(self):
        n = 0
        #pi_n =   np.random.choice(np.array(["L","R"]), self.n_states) # Initialise policies as random
        #pi_np1 = np.random.choice(np.array(["L","R"]), self.n_states)
        
        #Initialize PI w. speedup
        pi_n = np.repeat("R", self.n_states)
        pi_n[0] = "L"
        pi_np1 = np.repeat("R", self.n_states)

        if np.all(pi_n == pi_np1):
            self.policy_iteration()

        all_R = self.gen_matrices(np.repeat("R", self.n_states))
        all_L = self.gen_matrices(np.repeat("L", self.n_states))

        while np.any(pi_n != pi_np1):
            pi_n = pi_np1
            V_n = self.gen_matrices(pi_n)["value"] #Find value of pi_n
            # Generate rewards for every action in each state

            V_np1_R = all_R["reward"] + self.gamma*all_R["transition"]@V_n
            V_np1_L = all_L["reward"] + self.gamma*all_L["transition"]@V_n
            V_np1 = np.maximum(V_np1_R,V_np1_L) # Get elementwise max
            pi_np1 = np.where(V_np1_R == V_np1, "R","L") # Get argmax
            n = n+1

        self.n_iter_PI = n
        self.final_policy_PI = pi_np1
        self.final_valuef_PI = V_np1
        Q_star = np.empty((self.n_states,2))
        Q_star[:,0] = all_L["reward"] + self.gamma * (all_L["transition"] @ self.final_valuef_PI)
        Q_star[:,1] = all_R["reward"] + self.gamma * (all_R["transition"] @ self.final_valuef_PI)
        self.Q_star = Q_star

    def q_learning(self, alpha = "constant", incremental = False ,rounds = 10**5, p_go_right=np.array([0.5,0.5,0.5,0.5])):
        # Initialize q and reset all nodes
        self.reset_all()
        q_t = np.zeros((rounds, self.n_states, 2))

        #Convience translator between left and right and 0/1 for referencing numpy arrays
        action_dict = {"L":0,"R":1}
        reverse_dict = {"0":"L","1":"R"}

        # Specify learning rates
        if alpha == "constant":
            learning_rate = np.arange(rounds)+1
            learning_rate = 2/(learning_rate**(2/3)+1)
        else:
            learning_rate = np.zeros(rounds)

        # Specifies whether learning should be inceremental, and if so, sets B_t
        if incremental:
            B_t = np.zeros((rounds, self.n_states, 2))

        for i in range(rounds):
            if i == 0:
                index = 0
            
            # Updates all q-values (and also B-values) to that of the previous round - done here because of index convinience
            q_t[i,:,:] = q_t[i-1,:,:]
            if incremental:
                B_t[i,:,:] = B_t[i-1,:,:]
            
            # If not incremental sample randomly according to prescribed policy
            if not incremental:
                action_sample = np.random.choice(["L","R"], size = 1, p = [1-p_go_right[index], p_go_right[index]])[0]
            # If incremental, take the greedy action w.r.t. Q + B
            if incremental:
                action_sample = np.argmax(q_t[i,index,:]+B_t[i,index,:])
                action_sample = reverse_dict[str(action_sample)]
            # If learning is adpative update step-size parameter
            if alpha == "adaptive":
                learning_rate[i] = 2/(self.nodes[index].actions_taken[action_sample]**(2/3)+1)


            next,reward = self.nodes[index].visit(action_sample)

            delta = reward + self.gamma*np.max(q_t[i-1,index+next,:]) - q_t[i-1,index,action_dict[action_sample]]

            q_t[i,index,action_dict[action_sample]] += learning_rate[i]*delta
            
            if incremental:
            # Update B_t 
                for action in ["L","R"]:
                    if action == action_sample:
                        B_t[i,index,action_dict[action]] = 0
                    else:
                        B_t[i,index,action_dict[action]] += 1/self.nodes[index].times_since_last_play[action]
            index += next
        return q_t

class PrimitivePolicy:
    def __init__(self, environment, actions):
        """
        Initizialise policy on set of actions
        ----------
        Parameters
        ----------
        environment : Object
            instance of riwerswim class
        
        actions : List
            List of same length as the riwerswim environment. The i'th entry of the list should be a dict of the form {"L":a,"R":1-a} for a in the 
            interval between 0 and 1. The dict specifies the actions being taken with what probability in each state
        
        """
        self.environment = environment
        self.actions = actions 
        if self.environment.n_states != len(self.actions):
            raise Exception("Length of policy does not match the number of states")
        
    def execute_policy(self):
        """
        Executes policy on the current state of the environment
        ----------
        Parameters:
        ----------
        """
        action_sample = np.random.choice(a = list(self.actions[self.environment.current_state].keys()) , size = 1 , p=list(self.actions[self.environment.current_state].values()))[0]# Sample actions
        next, reward = self.environment.nodes[self.environment.current_state].visit(action_sample) # Visit next round
        self.environment.current_state += next #

class Option:
    def __init__(self, init_set, policy, termination_probs):
        """
        Initializes option on the environment
        ----------
        Parameters
        ----------
        init_set : list
            List of states where the option can be executed
        policy : instance of PrimitivePolicy class
            Policy to follow during execution of option
        termination_probabilities: list
            List containing the probabilities of the option executing in each state
        """
        self.init_set = init_set
        self.policy = policy
        self.termination_probs = termination_probs
    
    def execute_option(self):
        terminated = False
        while not terminated:
            self.policy.execute_policy()
            terminated = np.random.binomial(1, p = self.termination_probs[self.policy.environment.current_state])