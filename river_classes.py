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
    """
    Riverswim environment
    -----------
    Parameters
    -----------
    gamma : float between 0 and 1
        Discount factor
    nodes : List of instances of RiverNodes
        Defaults to standardriverswim
    n_states : int
        Number of states. Not necessary if nodes supplied explicitly. If nodes not supplied uses n_states as constructor of nodes.
    other parameters : Rewards/transition probs for all states
    """
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
        self.gen_all() # Gens matrices
        self.reset_all() # Resets all nodes
    
    def reset_all(self):
        """
        Resets all nodes
        """
        for node in self.nodes:
            node.reset_node()
            self.current_state = 0 


    def gen_matrices(self, policy):
        """
        Method for generating reward, transition matrix and value vector for a given policy. 
        Deprecated and essentially only used for generating the all-left and all-right matrices
        ----------
        Parameters
        ----------
        policy : policy on the form ['L','R','R', ...]
        ----------
        Returns
        ----------
        dict of rewards, transtion and value
        """
        reward_vector = np.array([self.nodes[i].rewards[policy[i]] for i in range(self.n_states)]) # Generates 

        transitionmatrix = np.zeros((self.n_states+1, self.n_states+1)) # Initialise 0-matrix for transition probabilities. Pad edges with +1
        
        for i in range(self.n_states):
            transitionmatrix[i, [i-1,i,i+1]] = self.nodes[i].transition_probs[policy[i]] # Fill out transition_matrix
        
        transitionmatrix = np.array(transitionmatrix)[0:self.n_states,0:self.n_states] # Make a numpy-array and throw padding away
        
        value_vector = np.linalg.inv(np.identity(self.n_states)-self.gamma*transitionmatrix) @ reward_vector # Calculate value of policy

        return {"reward":reward_vector, "transition":transitionmatrix, "value":value_vector} # Output dict containing reward vektor, transition matrix and value vector
    
    # Generates transition/reward of all_L and all_R policies. All other (stationary) policies can be derived from taking a combination of these
    def gen_all(self):
        """
        Method for generating all_L and all_R matrices. Called in __init__
        """
        self.all_R = self.gen_matrices(np.repeat("R", self.n_states))
        self.all_L = self.gen_matrices(np.repeat("L", self.n_states))



    #Does the same as gen_matrices, but for stochastic policies. Should be called after gen_all()
    #def eval_stochastic_policy(self, p_go_right=np.array([0.5,0.5,0.5,0.5])):
        #reward_vector = self.all_R["reward"] * p_go_right + self.all_L["reward"]*(1-p_go_right)
        #transition_matrix = self.all_R["transition"]*p_go_right + self.all_L["transition"]*(1-p_go_right)
        #value_vector = np.linalg.inv(np.identity(self.n_states)-self.gamma*transition_matrix) @ reward_vector # Calculate value of policy
        #return {"reward":reward_vector, "transition":transition_matrix, "value":value_vector} # Output dict containing reward vektor, transition matrix and value vector

    def policy_iteration(self):
        n = 0
        #Initialize PI w. speedup
        pi_n = np.repeat("R", self.n_states)
        pi_n[0] = "L"
        pi_np1 = np.repeat("R", self.n_states)

        if np.all(pi_n == pi_np1):
            self.policy_iteration()

        while np.any(pi_n != pi_np1):
            pi_n = pi_np1
            V_n = self.gen_matrices(pi_n)["value"] #Find value of pi_n
            # Generate rewards for every action in each state

            V_np1_R = self.all_R["reward"] + self.gamma*self.all_R["transition"]@V_n
            V_np1_L = self.all_L["reward"] + self.gamma*self.all_L["transition"]@V_n
            V_np1 = np.maximum(V_np1_R,V_np1_L) # Get elementwise max
            pi_np1 = np.where(V_np1_R == V_np1, "R","L") # Get argmax
            n = n+1

        self.n_iter_PI = n
        self.final_policy_PI = pi_np1
        self.final_valuef_PI = V_np1
        Q_star = np.empty((self.n_states,2))
        Q_star[:,0] = self.all_L["reward"] + self.gamma * (self.all_L["transition"] @ self.final_valuef_PI)
        Q_star[:,1] = self.all_R["reward"] + self.gamma * (self.all_R["transition"] @ self.final_valuef_PI)
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
    
    def vanilla_q_learning(self, exploration_policy, step_size, step_size_func,rounds):
        """
        Vanilla q-learning
        ----------------
        Parameters
        ----------------
        exploration_policy : instance of policy class
            The policy to be used for the data-generating process in q-learning
        step_size : str
            The step-size to use for q-learning choose either adaptive or constant
        step_size_func : function
            function that goes from the reals to the reals. Used for calculating the stepsize
        rounds : int
            Number of rounds to do q-learning for
        """

        # Initialize q and reset all nodes
        self.reset_all()
        q_t = np.zeros((rounds, self.n_states, 2))

        #Convience translator between left and right and 0/1 for referencing numpy arrays
        action_dict = {"L":0,"R":1}
        reverse_dict = {"0":"L","1":"R"}

        # Specify learning rates
        if step_size == "constant":
            learning_rate = np.arange(rounds)+1
            learning_rate = step_size_func(learning_rate)
        else:
            learning_rate = np.zeros(rounds)

        for i in range(rounds):
            # Updates all q-values - done here because of index convinience
            q_t[i,:,:] = q_t[i-1,:,:]
            #Update the adaptive stepsize
            curr_state = self.current_state
            
            # execute action according to exploration policy. Note - updates self.current_state 
            action_sample, reward = exploration_policy.execute_policy(self)
            if step_size == "adaptive":
                learning_rate[i]  = step_size_func(self.nodes[curr_state].actions_taken[action_sample])

            delta = reward + self.gamma*np.max(q_t[i-1,self.current_state,:]) - q_t[i-1,curr_state,action_dict[action_sample]]

            q_t[i,curr_state,action_dict[action_sample]] += learning_rate[i]*delta
        return q_t
    def option_q_learning(self, exploration_policy, step_size, step_size_func)


class PrimitivePolicy:
    def __init__(self, actions):
        """
        Initizialise policy on set of actions
        ----------
        Parameters
        ----------
        actions : List
            List of same length as the riwerswim environment. The i'th entry of the list should be a dict of the form {"L":a,"R":1-a} for a in the 
            interval between 0 and 1. The dict specifies the actions being taken with what probability in each state
        
        """
        self.actions = actions 
        
    def execute_policy(self,environment):
        """
        Executes policy on the current state of the environment
        ----------
        Parameters:
        ----------
        environment : Object
            instance of riwerswim class
        ----------
        Returns:
        ----------
        Returns the reward of the taken action and updates RiwerSwim environment.
        """
        if len(self.actions) != environment.n_states:
            raise Exception("Length of policy does not match the number of states")
         
        action_sample = np.random.choice(a = list(self.actions[environment.current_state].keys()) , size = 1 , p=list(self.actions[environment.current_state].values()))[0]# Sample actions
        next, reward = environment.nodes[environment.current_state].visit(action_sample) # Visit next round
        environment.current_state += next #
        return action_sample,reward

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
    
    def execute_option(self,environment):
        """
        executes option on the environment
        ----------
        Parameters
        ----------
        Environment: instance of RiverSwim class
        ----------
        Returns
        ----------
        Nothing. updates riverswim environment
        """    
        terminated = False
        while not terminated:
            self.policy.execute_policy(environment) # Execute policy being followed on the environment
            terminated = np.random.binomial(1, p = self.termination_probs[environment.current_state]) # Sample whether to terminate
