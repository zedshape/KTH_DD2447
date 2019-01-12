from q2generator import *
import matplotlib.pyplot as plt
import numpy as np
import pdb
import math
import random

def conditional_likelihood(o, G, start, p, X): 
	states = []
	entries = []
	states.append(start)
	entries.append(0)				#entry directions

	# initial state positions
	s1_r = states[0].row
	s1_c = states[0].col 

	T = len(o)		#number of observations
	no_of_edges = len(start.edges)
	no_of_positions = G.lattice_size **2
	lattice_dim = (G.lattice_size, G.lattice_size)
	matrix_dim = (2, no_of_positions, no_of_positions)

	#compute transition probability matrix
	t_prob = np.zeros(matrix_dim)  

	counter = 0
	for i in range(G.lattice_size):
		for j in range(G.lattice_size):	
			cur_node = G.get_node(i,j)		# current node
			cur_ss = X[i][j]				# current switch state
			edges = cur_node.edges 			# edges of current node
			exit_edges = (cur_node.get_edge(0), cur_node.get_edge(cur_ss))		# can exit through zero or switch state

			for e in range(len(exit_edges)):		# possible next nodes
				i_next, j_next = validate_idx(i + exit_edges[e][0], j + exit_edges[e][1], G.lattice_size)		#index of next node
				next_node = G.get_node(i_next, j_next)

				first_index = np.ravel_multi_index((i,j),lattice_dim)
				second_index = np.ravel_multi_index((i_next,j_next),lattice_dim)

				t_prob[e][first_index][second_index] = 1			# e = 0 if zero exit (entry from {1,2,3} ), e=1 if switch state exit (entry from 0)

	# computing the path, using start position and transition probability matrix
	e1 = states[-1].get_edge(X[s1_r][s1_c])		#starts with following switch state
	r,c = validate_idx(s1_r + e1[0],s1_c + e1[1], G.lattice_size)
	states.append(G.get_node(r,c))	#second state
	entries.append(G.get_entry_direction(states[-2], states[-1]))

	for t in range(2,T): 
		e_dir = G.get_entry_direction(states[-2], states[-1])
		entries.append(e_dir)

		r = states[-1].row
		c = states[-1].col

		i_long = np.ravel_multi_index((r,c), lattice_dim)

		if e_dir > 0:
			i = np.unravel_index(t_prob[0][i_long].argmax(), lattice_dim)		# use switch state
		else:
			i = np.unravel_index(t_prob[1][i_long].argmax(), lattice_dim)		# exit from zero

		states.append(G.get_node(i[0],i[1]))

	#compute emission probability matrix
	e_prob = np.zeros(((2, no_of_positions, len(edges))))  # t_prob[entry][position][observation]

	for pos in range(no_of_positions):
		r,c = np.unravel_index(pos,lattice_dim)	# row and col indices
		ss_t = X[r][c]						# switch state exit
		
		# probability of incorrect observation is p, thus probability of a specific incorrect observation is p/3  (--> total obs prob is always one)
		e_prob[0][pos] = [p/3 for i in range(no_of_edges)]
		e_prob[1][pos] = [p/3 for i in range(no_of_edges)]

		# probability of correct observation *overwriting*
		e_prob[0][pos][0] = 1-p   #if it enters current state from {1,2,3}, the output should be 0
		e_prob[1][pos][ss_t] = 1-p	#if it enters current state from {0}, the output should be the switch state

	#calculate conditional probability
	cond_prob = 0
	for t in range(1,T):			# normalize?
		statesum = 0
		i1 = np.ravel_multi_index((states[t-1].row,states[t-1].col),lattice_dim)  #index of s(t-1)
			
		if entries[t-1] >0:	#st-1
			prev_out = 0
		else:
			prev_out = 1

		#for st in range(no_of_positions):
		i2 = np.ravel_multi_index((states[t].row,states[t].col),lattice_dim)	#index of s(t)

		if entries[t] > 0:		# if entering s(t-1) from direction{1,2,3}, path will follow zero-direction to s(t)
			cur_out = 0		
		else:						#if entering s(t-1) from zero-direction, path will follow switch state to s(t)
			cur_out = 1		

		epdf = e_prob[cur_out][i2][(X[states[t].row][states[t].col])]
		tpdf = t_prob[prev_out][i1][i2]
		statesum += epdf * tpdf

		if statesum != 0:
			cond_prob += math.log(statesum)
			#print(cond_prob)
	return(cond_prob)	

# o: observations
# n_lattice: size of the lattice
# num_iter: number of MCMC iterations

def mh_w_gibbs(o, G, num_iter, error_prob=0.1):			#metropolis-hastings
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

	#s.append(G.get_node(1,0))
	probabilities = []
	probabilities.append(conditional_likelihood(o,G,s[0],error_prob, X[0]))

	#variables created
	lattice_dim = (G.lattice_size, G.lattice_size)		#lattice dimension, used for index conversion
	no_of_positions = G.lattice_size **2		# number of possible positions
	no_of_ss = G.lattice_size		# number of possible switch states {1,2,3}

	X_count = np.ones((no_of_positions, len(X[0])))		# counts occurances of each switch state at each position
	s_count = np.ones(no_of_positions)			# counts number of times each index is start position

	accepted_n = []
	accepted_n.append(0)

	X_prev = X[0]		# initial sample of X
	s_prev = s[0]
	s_prev_index = np.ravel_multi_index((s[0].row,s[0].col),lattice_dim)  # index of initial sample of start position

	for n in range(num_iter): #change back to num_iter
		# count occurances in previous sample
		#s_count[s_prev_index] += 1						#add count
		#s_count_n = s_count * 1/np.sum(s_count)		#normalize
		#s_new_index = np.nonzero(np.random.multinomial(1, s_count_n , size=1))[1][0]	# sample from categorical distribution
		
		#s_new_index = np.random.randint(G.lattice_size**2, size = 1)[0]
		#print(s_new_index)
		#s_r,s_c = np.unravel_index(s_new_index, lattice_dim)	
		#s_new = G.get_node(s_r, s_c)

		for x_row in range(len(X_prev)):		# count occurances of every switch state at every position
			for x_col in range(len(X_prev)):
				X_i_long = np.ravel_multi_index((x_row,x_col),lattice_dim)
				X_count[X_i_long][(X_prev[x_row][x_col])-1] += 1	# index 0 --> ss=1 etc
		
		X_count_normalized = [X_count[i][:] * 1/np.sum(X_count[i][:]) for i in range(no_of_positions)]			#normalize
		
		X_new = np.zeros(np.shape(X_prev))

		for x in range(no_of_positions):				# creating new sample
			#sample_ss_index = np.random.multinomial(1, X_count_normalized[x], size = 1)
			#sample_ss = np.nonzero(sample_ss_index)[1][0]
			sample_ss = np.random.randint(no_of_ss)
			r,c = np.unravel_index(x, lattice_dim)
			X_new[r][c] = int(sample_ss+1 )	# ss index 0 --> ss=1 etc

		X_new = X_new.astype(int)

		# test
		X_new = X_prev
		ss_to_change = np.random.randint(no_of_positions)
		r,c = np.unravel_index(ss_to_change, lattice_dim)
		X_new[r][c] = np.random.randint(no_of_ss)
		# end test

		probs = []
		for i in range(no_of_positions):
			r,c = np.unravel_index(i, lattice_dim)
			probs.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_new))			# append prob for every start position

		#probs_normalized = probs/np.sum(probs)
		most_prob_s = np.argmax(probs)
	

		#s_new_index = np.nonzero(np.random.multinomial(1, probs_normalized, size=1))[1][0]		#test normal distr
		#new_prob = probs[s_new_index]
		s_r,s_c = np.unravel_index(most_prob_s, lattice_dim)
		s_new = G.get_node(s_r, s_c)

		new_prob = probs[np.argmax(probs)]
		old_prob = conditional_likelihood(o,G,s_prev,error_prob, X_prev) 
		#new_prob = conditional_likelihood(o,G,s_new,error_prob, X_new)
		ratio = new_prob/old_prob

		print(ratio)
		if ratio <0:
			print("_______ERROR!!!___________")
		r = min(ratio, 1)

		u = np.random.uniform(0.75,1)
		if u < r:		#accept proposal
			print("accept")
			X.append(X_new)
			X_prev = X_new
			s.append(s_new)
			s_prev = s_new
			s_prev_index = most_prob_s
			probabilities.append(new_prob)
			accepted_n.append(n)

	print("Accepted: " + str(len(probabilities)))
	print(np.argmax(probabilities))
	print(probabilities[np.argmax(probabilities)])
	print(math.exp(max(probabilities)))
	print(s[np.argmax(probabilities)])

	s_index = [np.ravel_multi_index((s[i].row,s[i].col),lattice_dim) for i in range(len(s))]
	X_test = [X[i][2][1] for i in range(len(X))]

	#plt.plot(accepted_n, s_index)
	#plt.show()

	# the histogram of the data
	#plt.figure(1)
	plt.subplot(311)
	n, bins, patches = plt.hist(s_index, 9)
	plt.axis([0, 9, 0, 500])
	plt.xlabel('Start position')
	plt.ylabel('Accepted samples')
	plt.title('Histogram of possible start positions')
	plt.grid(True)

	#plt.figure(2)
	plt.subplot(312)

	n, bins, patches = plt.hist(X_test, 3, facecolor='g')

	plt.xlabel('(1,1) switch')
	plt.ylabel('Accepted samples')
	plt.title('Histogram of possible switches for (1,1)')
	plt.axis([0, 3, 0, 1000])

	plt.grid(True)


	plt.subplot(313)
	plt.axis([0, 1000, 0, 9])
	plt.plot(accepted_n, s_index)
	plt.show()

	return s, X

def gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]


	probabilities = []
	probabilities.append(conditional_likelihood(o,G,s[0],error_prob, X[0]))

	#variables created
	lattice_dim = (G.lattice_size, G.lattice_size)		#lattice dimension, used for index conversion
	no_of_positions = G.lattice_size **2		# number of possible positions
	no_of_ss = G.lattice_size		# number of possible switch states {1,2,3}

	X_count = np.ones((no_of_positions, len(X[0])))		# counts occurances of each switch state at each position
	s_count = np.ones(no_of_positions)			# counts number of times each index is start position

	accepted_n = []
	accepted_n.append(0)

	X_prev = X[0]		# initial sample of X
	s_prev = s[0]
	s_prev_index = np.ravel_multi_index((s[0].row,s[0].col),lattice_dim)  # index of initial sample of start position

	for n in range(num_iter):  #num_iter
		print("n: " + str(n))

		X_new = X_prev
		ss_to_change = n % no_of_positions
		#ss_to_change = np.random.randint(no_of_positions)
		r,c = np.unravel_index(ss_to_change, lattice_dim)
		X_new[r][c] = np.random.randint(no_of_ss)

		probs = []
		for i in range(no_of_positions):
			r,c = np.unravel_index(i, lattice_dim)
			probs.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_new))			# append prob for every start position

		probs_normalized = probs/np.sum(probs)
		most_prob_s = np.argmax(probs)
	

		#s_new_index = np.nonzero(np.random.multinomial(1, probs_normalized, size=1))[1][0]		#test normal distr
		#new_prob = probs[s_new_index]
		s_r,s_c = np.unravel_index(most_prob_s, lattice_dim)
		s_new = G.get_node(s_r, s_c)

		new_prob = probs[np.argmax(probs)]
		old_prob = conditional_likelihood(o,G,s_prev,error_prob, X_prev) 
		#new_prob = conditional_likelihood(o,G,s_new,error_prob, X_new)
		ratio = new_prob/old_prob

		print(ratio)
		r = min(ratio, 1)

		u = np.random.uniform(0,1)
		if u < r:		#accept proposal
			print("accept")
			X.append(X_new)
			X_prev = X_new
			s.append(s_new)
			s_prev = s_new
			s_prev_index = most_prob_s
			probabilities.append(new_prob)
			accepted_n.append(n)

	print("Accepted: " + str(len(probabilities)))
	print(np.argmax(probabilities))
	print(probabilities[np.argmax(probabilities)])
	print(math.exp(max(probabilities)))
	print(s[np.argmax(probabilities)])

	s_index = [np.ravel_multi_index((s[i].row,s[i].col),lattice_dim) for i in range(len(s))]
	X_test = [X[i][2][1] + 1 for i in range(len(X))]

	#plt.plot(accepted_n, s_index)
	#plt.show()

	# the histogram of the data
	#plt.figure(1)
	plt.subplot(311)
	n, bins, patches = plt.hist(s_index, 9)
	plt.axis([0, 9, 0, 500])
	plt.xlabel('Start position')
	plt.ylabel('Accepted samples')
	plt.title('Histogram of possible start positions')
	plt.grid(True)

	#plt.figure(2)
	plt.subplot(312)

	n, bins, patches = plt.hist(X_test, 3, facecolor='g')

	plt.xlabel('(1,1) switch')
	plt.ylabel('Accepted samples')
	plt.title('Histogram of possible switches for (1,1)')
	plt.axis([0, 3, 0, 1000])

	plt.grid(True)


	plt.subplot(313)
	plt.axis([0, 1000, 0, 9])
	plt.plot(accepted_n, s_index)
	plt.show()

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

	#s2, X2 = gibbs(o, G, num_iter, p)
	

	# YOUR CODE:
	# analyze s, X by comparing to the ground truth
	# check for convergence

	#conditional_likelihood(o,G,s,p)

if __name__ == '__main__':
	main()


