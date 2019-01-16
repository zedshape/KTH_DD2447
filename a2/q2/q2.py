from q2generator import *
import matplotlib.pyplot as plt
import numpy as np
import pdb
#manually added
import pandas as pd
from itertools import permutations

def conditional_likelihood(o, G, start, p, X):
    states = []
    entries = []				#entry directions

    lattice_dim = (G.lattice_size, G.lattice_size)
    T = len(o)#number of observations

    # initial state positions
    s1_r = start.row
    s1_c = start.col
    s1 = G.get_node(s1_r, s1_c) #start position node
    states.append(s1)

    #get second position to check transition
    states.append(G.get_next_node(states[0], 0, X)[0]) #next node

    cond_prob = 1

    #for looping timetable
    for t in range(1, T):
        #check the transition entry
        #print(states[-2], states[-1])
        e_dir = G.get_entry_direction(states[-2], states[-1])
        entries.append(e_dir)

        if e_dir > 0:
            if o[t] == 0:
                cond_prob = cond_prob * (1-p)
            else:
                cond_prob = cond_prob * p
        elif e_dir == 0:
            #ot == dt -> emission probability
            #retrieve t'th data and compare
            dt = X[states[-1].row][states[-1].col]
            if o[t] == dt:
                cond_prob = cond_prob * (1-p)
            else:
                cond_prob = cond_prob * p

        #update next position to loop until timetable ends
        states.append(G.get_next_node(states[-1], e_dir, X)[0])

    return(cond_prob)

# o: observations
# n_lattice: size of the lattice
# num_iter: number of MCMC iterations

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
    total_trial = 0

    X_prev = X[0]		# initial sample of X
    s_prev = s[0]
    #s_prev_index = np.ravel_multi_index((s[0].row,s[0].col),lattice_dim)  # index of initial sample of start position

    for n in range(num_iter): #change back to num_iter
        X_prev = X[-1]
        #print(X_prev)
        probs = []
        for i in range(no_of_positions):
            r,c = np.unravel_index(i, lattice_dim)
            probs.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_prev))			# append prob for every start position

        #for further iteration
        probs = np.array(probs)
        probs = probs/np.sum(probs) #NORMALIZATION

        #############gibbs update without acceptance process#############
        #print(probs)
        #multinomial proposal
        most_prob_s = np.argmax(np.random.multinomial(1,probs))
        #print(most_prob_s)
        #for plotting
        probabilities.append(probs[most_prob_s])
        #get new s based on our proposal
        s_r,s_c = np.unravel_index(most_prob_s, lattice_dim)
        s_new = G.get_node(s_r, s_c)
        #s_prev_index = most_prob_s
        s.append(s_new)

        #############MH update with acceptance process#############
        #proposal of X_new
        #print(s_r, s_c)

        #we have X_prev and X_new as 3*3
        #this value will be filled in iteration
        X_new = np.zeros(np.shape(X_prev), dtype=int)
        X_prop = X_prev.copy()

        #we need to perform it for all v's in v=no_of_positions
        for x in range(no_of_positions):				# creating new sample
            sample_ss = np.random.randint(no_of_ss)
            #keep index for update
            r,c = np.unravel_index(x, lattice_dim)

            new = np.random.randint(1, no_of_ss+1) #from 1 to no_of_ss
            old = int(X_prev[r][c])

            X_prop[r][c] = new

            #propose with s_new value
            old_prob = conditional_likelihood(o,G,s_new,error_prob, X_prev)
            new_prob = conditional_likelihood(o,G,s_new,error_prob, X_prop)
            ratio = new_prob/old_prob

            rate = min(ratio, 1)
            u = np.random.uniform(0,1)
            total_trial += 1
            if u < rate:		#accept proposal
                X_prev = X_prop.copy() #update X
                X_new[r][c] = new #new value update
                old_prob = new_prob
                accepted_n.append(n)
            else:
                X_prop = X_prev.copy()
                X_new[r][c] = old


        #we keep old X if X is not accepted, update new X if accepted
        X.append(X_new)

    print("Total acceptance rate for MH: ", len(accepted_n)/total_trial)

    return s, X


def gibbs(o, G, num_iter, error_prob=0.1):
    s = [] # store samples for the start positions
    X = [] # store switch states
    X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
    s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

    #burn in rate
    burn_in_rate = 0.4
    burn_in_num = int(burn_in_rate * num_iter)

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

    X_prev = np.array(X[0])		# initial sample of X
    s_prev = s[0]
    #s_prev_index = np.ravel_multi_index((s[0].row,s[0].col),lattice_dim)  # index of initial sample of start position

    for n in range(num_iter + burn_in_num): #change back to num_iter
        X_prev = X[-1]
        #print(X_prev)
        probs = []
        for i in range(no_of_positions):
            r,c = np.unravel_index(i, lattice_dim)
            probs.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_prev))			# append prob for every start position

        #for further iteration
        probs = np.array(probs)
        probs = probs/np.sum(probs) #NORMALIZATION

        #############gibbs update without acceptance process#############
        #print(probs)
        #multinomial proposal
        most_prob_s = np.argmax(np.random.multinomial(1,probs))
        #print(most_prob_s)

        #get new s based on our proposal
        s_r,s_c = np.unravel_index(most_prob_s, lattice_dim)
        s_new = G.get_node(s_r, s_c)
        #s_prev_index = most_prob_s
        s.append(s_new)

        #now we do not need acceptance thing anymore
        X_new = np.copy(X_prev)
        pp = [] #to plot something... based on requirement

        #we need to perform it for all v's in v=no_of_positions
        for x in range(no_of_positions):				# creating new sample
            sample_ss = np.random.randint(no_of_ss)
            #keep index for update
            r,c = np.unravel_index(x, lattice_dim) #3**3
            tmp = []

            for new in range(1, no_of_ss+1): #from 1 to no_of_ss)
                #again multinomial sampling for each possible values
                X_new[r][c] = new
                tmp.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_new))

            #normalize
            new_prob = np.array(tmp)
            new_prob = new_prob/np.sum(new_prob)
            new = np.argmax(np.random.multinomial(1,new_prob))+1

            X_new[r][c] = new

            pp.append(new_prob)

        #we anyway update new X as it is gibbs sampling
        #print(X_new, X_prev)
        X.append(X_new)
        #print(X)
        probabilities.append(pp)

    #crop burn in phases
    s = s[burn_in_num:]
    X = X[burn_in_num:]

    return s, X

def block_gibbs(o, G, num_iter, error_prob=0.1):
    s = [] # store samples for the start positions
    X = [] # store switch states
    X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
    s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

    #copy and paste from our gibbs implementation for s1 part
    #burn in rate
    burn_in_rate = 0.4
    burn_in_num = int(burn_in_rate * num_iter)

    #s.append(G.get_node(1,0))
    probabilities = []
    #probabilities.append(conditional_likelihood(o,G,s[0],error_prob, X[0]))

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
    #s_prev_index = np.ravel_multi_index((s[0].row,s[0].col),lattice_dim)  # index of initial sample of start position

    loc = np.array([3,2,0,5,6,7,4,8,1])
    locs = np.split(loc, no_of_ss) #not it is three parts
    #print(locs)

    for n in range(num_iter + burn_in_num): #change back to num_iter
        X_prev = X[-1]
        #print(X_prev)
        probs = []
        for i in range(no_of_positions):
            r,c = np.unravel_index(i, lattice_dim)
            probs.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_prev))			# append prob for every start position

        #for further iteration
        probs = np.array(probs)
        probs = probs/np.sum(probs) #NORMALIZATION

        #############gibbs update without acceptance process#############
        #print(probs)
        #multinomial proposal
        most_prob_s = np.argmax(np.random.multinomial(1,probs))
        #print(most_prob_s)

        #get new s based on our proposal
        s_r,s_c = np.unravel_index(most_prob_s, lattice_dim)
        s_new = G.get_node(s_r, s_c)
        #s_prev_index = most_prob_s
        s.append(s_new)

        #now we do not need acceptance thing anymore
        X_new = np.copy(X_prev)

        ########## BLOCKED GIBBS PART ##########
        #we need to perform it for all v's in v=no_of_positions
        #randomly re-allocate location
        #loc = np.random.permutation(no_of_positions)
        #locs = np.split(loc, no_of_ss) #not it is three parts

        #generate possible combination of {1,2,3}
        #all
        perms = list(permutations([1, 2, 3]))

        #for one block each
        for l in locs:
            #block update candidates
            tmp = []
            #each possible permutation
            for perm in perms:
                #try to apply permutation to X value

                #three positions in a block
                for idx, ll in enumerate(l):
                    r,c = np.unravel_index(ll, lattice_dim) #3**3
                    #generate X_new with three value in current permutation
                    #print(r, c, idx)
                    X_new[r][c] = perm[idx]

                #now append probability of three value opdated
                #print(r, c)
                tmp.append(conditional_likelihood(o,G,G.get_node(r,c),error_prob, X_new))

            #normalize
            new_prob = np.array(tmp)
            new_prob = new_prob/np.sum(new_prob)
            new = np.argmax(np.random.multinomial(1,new_prob))

            target_perm = perms[new]
            #three positions in a block
            for idx, ll in enumerate(l):
                r,c = np.unravel_index(ll, lattice_dim) #3**3
                #generate X_new with three value in current permutation
                X_new[r][c] = target_perm[idx]

        ########################################
        #we anyway update new X as it is gibbs sampling
        X.append(X_new)

    #crop burn in phases
    s = s[burn_in_num:]
    X = X[burn_in_num:]

    return s, X


def diagnosis(R):
    #Gelman-Rubin diagnostic
    #reference: http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf

    n = R.shape[1] #number of iterations
    m = R.shape[0] #number of chains

    #calculate basic thetas
    theta_j = np.mean(R, axis=1)
    theta_bar = np.mean(theta_j, axis=0)

    #calculate W
    sj_square = 1/np.float(n-1)*np.sum(np.array([np.power((R[chain]-theta_j[chain]), 2) \
                                           for chain in range(m)]), axis=1)
    W = 1/np.float(m) * np.sum(sj_square, axis=0)
    #calculate B
    B = n/np.float(m-1)*np.sum(np.power(theta_j-theta_bar, 2), axis=0)

    #Finally get V value
    V_hat = (1-1/n)*W + 1/n*B

    #R
    R_hat = np.sqrt(V_hat/W)

    return R_hat

# generate sample graph and observations
def main():
	seed = 17
	n_lattice = 3		# --> we have 9 states in total
	T = 100				# 100 time steps
	p = 0.1				# probability of error
	G, X_truth, s_truth, o = generate_data(seed, n_lattice, T, p)   # nothing to be done here?
	print(o)				#MAKE NOT COMMENT
	print(X_truth)				#MAKE NOT COMMENT
	print(s_truth[1:20])		#MAKE NOT COMMENT

	# randomize the switch states -- get initial state
	X = sample_switch_states(n_lattice)

	# infer s[0] and switch states given o and G
	num_iter = 1000

	print("Calculating MH_Gibbs")
	s1, X1 = mh_w_gibbs(o, G, num_iter, p)
	print("Calculating Gibbs")
	s2, X2 = gibbs(o, G, num_iter, p)
	print("Calculating Blocked Gibbs")
	s3, X3 = block_gibbs(o, G, num_iter, p)

	# YOUR CODE:
	#####plotting area
    #We commented in all plt.show() for submission
    #Grader can comment them out to check our result

	s_lists = [s1, s2, s3]
	X_lists = [X1, X2, X3]
	labels = ["MH_Gibbs", "Gibbs", "Blocked"]

	# 1. plot histogram for each algorithm
	print("Plotting histograms")
	for idx in range(3):
	    lattice_dim = (G.lattice_size, G.lattice_size)
	    res = []

	    default_set = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

	    for ss in s_lists[idx]:
	        default_set[np.ravel_multi_index((ss.row, ss.col), lattice_dim)] += 1

	    res = pd.Series(default_set)

	    res.index = ['(0,0)','(0,1)','(0,2)','(1,0)','(1,1)','(1,2)','(2,0)','(2,1)','(2,2)']
	    res.plot(kind='bar', fontsize=20, figsize = (8,4), legend=False, title='Histogram of possible start positions for {}'.format(labels[idx]))

	#plt.show()

	#2. hit rate
	print("Calculating hit rate")
	X_truth_flat = np.array(X_truth).flatten()
	print("TRUTH: ", X_truth_flat)

	for idx in range(3):
		collection = []
		for i in X_lists[idx]:
			collection.append(np.array(i).flatten())
		collection = pd.DataFrame(collection)

		hit_rate = []
		probable = []
		for i in collection.columns:
		    hit_rate.append( np.sum(collection[i] == X_truth_flat[i])/len(collection[i]))
		    probable.append(collection[i].mode()[0])
		#get hit rate
		hit_rate = np.array(hit_rate)

		#get accuracy with most prevalent value
		accuracy = probable == X_truth_flat

		plt.plot(hit_rate, label=labels[idx])
		plt.legend()
		plt.axhline(y=1/3, color='r')
		plt.xticks(range(9), ['(0,0)','(0,1)','(0,2)','(1,0)','(1,1)','(1,2)','(2,0)','(2,1)','(2,2)'])
		plt.xlabel("labels")
		plt.ylabel("probability")

		print("======= {} =======".format(labels[idx]))
		print("Accuracy: ", accuracy)
		print("Hit rate: ", hit_rate)
		print("Prediction: ", probable)

	#plt.show()

	# 2. check for convergence
	# it takes a lot of time so that we comment them in
	print("Chekcing convergence (commented in)")
	"""
	multiple_models = [[mh_w_gibbs(o, G, num_iter, p) for _ in range(10)],
                   [gibbs(o, G, num_iter, p) for _ in range(10)],
                   [block_gibbs(o, G, num_iter, p) for _ in range(10)]]

	for idx, multiple_gibbs in enumerate(multiple_models):
	    gibbs_new = np.zeros([10, 1001, 10])
	    #each model
	    for idx_i, i in enumerate(multiple_gibbs):
	        #j = each value
	        for idx_j, j in enumerate(i):
	            #s
	            if idx_j == 0:
	                for idx_k, k in enumerate(j):
	                    gibbs_new[idx_i, idx_k, 0] = np.ravel_multi_index((k.row, k.col),lattice_dim)
	            else:
	                for idx_k, k in enumerate(j):
	                    val = np.array(k).flatten()
	                    for x, l in enumerate(val):
	                        #print(k)
	                        gibbs_new[idx_i, idx_k, x+1] = l
	    diagnosis_results = np.array([ diagnosis(gibbs_new[:,0:t,:]) for t in range(2,num_iter)])
	    plt.figure(figsize=(8,6))
	    plt.title('Diagnosis result(Convergence rate) using Gelman-Rubin diagnostic for {}'.format(labels[idx]))
	    for n in range(10): # for each position plot a new graph
	        if n == 0:
	            plt.plot(diagnosis_results[:,n], label='s_1')
	        else:
	            r,c = np.unravel_index(n-1, (3,3))
	            plt.plot(diagnosis_results[:,n], label='x[{},{}]'.format(r, c))
	    plt.legend(loc=1)
	plt.show()
	"""

if __name__ == '__main__':
	main()
