from q2generator import *
import numpy as np
import pdb

def conditional_likelihood(o, G, start):
	pass

# o: observations
# n_lattice: size of the lattice
# num_iter: number of MCMC iterations
def mh_w_gibbs(o, G, num_iter, error_prob=0.1):
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

def block_gibbs(o, G, num_iter, error_prob=0.1):
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
	n_lattice = 3
	T = 100
	p = 0.1
	G, X_truth, s_truth, o = generate_data(seed, n_lattice, T, p)
	print(o)
	print(X_truth)
	print(s_truth[1:20])

	# randomize the switch states -- get initial state
	X = sample_switch_states(n_lattice)

	# infer s[0] and switch states given o and G
	num_iter = 1000
	s, X = mh_w_gibbs(o, G, num_iter, p)
	
	# YOUR CODE:
	# analyze s, X by comparing to the ground truth
	# check for convergence

if __name__ == '__main__':
	main()