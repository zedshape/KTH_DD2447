from q2generator import *
import numpy as np
import pdb
import math

def conditional_likelihood(o, G, start, p): 
	states = []
	entries = []
	states.append(start[0])
	entries.append(0)				#entry directions

	s1_r = states[0].row
	s1_c = states[0].col #initial positions

	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state

	T = len(o)		#number of observations
	S = G.lattice_size **2 	# of positions

	lattice_dim = (G.lattice_size, G.lattice_size)
	matrix_dim = (2, G.lattice_size**2, G.lattice_size**2)

	#compute transition probability matrix
	t_prob = np.zeros(matrix_dim)  

	counter = 0
	for i in range(G.lattice_size):
		for j in range(G.lattice_size):	
			cur_node = G.get_node(i,j)		# current node
			cur_ss = X[0][i][j]				# current switch state

			edges = cur_node.edges 			# edges of current node
			exit_edges = (cur_node.get_edge(0), cur_node.get_edge(cur_ss))		# can exit through zero or switch state
			#print(exit_edges)
			if i == 2:
				if j== 0:
					print("2,0")
					print(G.get_node(i,j))
					print(G.get_node(i,j).edges)

			for e in range(len(exit_edges)):
				i_next, j_next = validate_idx(i + exit_edges[e][0], j + exit_edges[e][1], G.lattice_size)		#index of next node
				
				next_node = G.get_node(i_next, j_next)

				first_index = np.ravel_multi_index((i,j),lattice_dim)
				second_index = np.ravel_multi_index((i_next,j_next),lattice_dim)

				#print("________________")
				#print("e " + str(e))
				#print("current_node: ")
				#print(cur_node)

				#print("next_node: ")
				#print(next_node)

				#print("Current node switch state: " + str(cur_ss))

				t_prob[e][first_index][second_index] = 1			# e = 0 if zero exit, e=1 if switch state exit

	print(t_prob)
	print(X[0])

	# computing the path, using start position and transition probability matrix

	e1 = states[-1].get_edge(X[0][s1_r][s1_c])		#starts with following switch state
	states.append(G.get_node(s1_r + e1[0], s1_c + e1[1]))		#second state
	entries.append(G.get_entry_direction(states[-2], states[-1]))

	for t in range(2,T): 

		e_dir = G.get_entry_direction(states[-2], states[-1])
		entries.append(e_dir)

		r = states[-1].row
		c = states[-1].col

		i_long = np.ravel_multi_index((r,c), lattice_dim)

		if e_dir > 0:
			i = np.unravel_index(t_prob[0][i_long].argmax(), lattice_dim)		# use switch state
			#print("zero-matrix")
		else:
			i = np.unravel_index(t_prob[1][i_long].argmax(), lattice_dim)		# exit from zero
			#print("ss-matrix")
		states.append(G.get_node(i[0],i[1]))

	#compute emission probability matrix
	e_prob = np.zeros(((G.lattice_size**2, len(edges))))  # t_prob[obs_i][obs_j][i][j]

	for t in range(T):
		r = states[t].row
		c = states[t].col
		ss_t = X[0][r][c]

		i = np.ravel_multi_index((r,c),lattice_dim)

		if entries[t] == 0:
			if o[t] == ss_t:
				print("match")
				e_prob[i][ss_t] = 1-p
			else:
				e_prob[i][ss_t] = p

	cond_prob = 0

	for t in range(1,T):
		statesum = 0
		for st in range(len(states)):
			i1 = np.ravel_multi_index((states[st-1].row,states[st-1].col),lattice_dim)
			i2 = np.ravel_multi_index((states[st].row,states[st].col),lattice_dim)

			if entries[st] >0:
				output_matrix = 0
			else:
				output_matrix = 1

			statesum += (e_prob[i2][(X[0][states[st].row][states[st].col])]) * t_prob[output_matrix][i1][i2]
		cond_prob += math.log(statesum)
		print(cond_prob)	

	print(cond_prob)
	print(math.exp(cond_prob))


# o: observations
# n_lattice: size of the lattice
# num_iter: number of MCMC iterations
def mh_w_gibbs(o, G, num_iter, error_prob=0.1):			#metropolis-hastings
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
	
	for n in range(num_iter):

		pass


	return s, X

def gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

	
	for n in range(num_iter):


		pass
	return s, X

def block_gibbs(o, G, num_iter, error_prob=0.1):		#blocked gibbs sampling
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]
	for n in range(num_iter):
		pass
	return s, X

# generate sample graph and observations
def main():
	seed = 17
	n_lattice = 3		# --> we have 9 states in total
	T = 100				# 100 time steps
	p = 0.1				# probability of error
	G, X_truth, s_truth, o = generate_data(seed, n_lattice, T, p)   # nothing to be done here?
	print(o)				#MAKE NOT COMMENT
	#print(X_truth)				MAKE NOT COMMENT
	#print(s_truth[1:20])		MAKE NOT COMMENT


	# randomize the switch states -- get initial state
	X = sample_switch_states(n_lattice)

	# infer s[0] and switch states given o and G
	num_iter = 1000
	s, X = mh_w_gibbs(o, G, num_iter, p)
	

	# YOUR CODE:
	# analyze s, X by comparing to the ground truth
	# check for convergence

	conditional_likelihood(o,G,s,p)

if __name__ == '__main__':
	main()


