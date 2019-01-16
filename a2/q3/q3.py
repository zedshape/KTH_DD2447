#for safe division: it is option, not library
from __future__ import division
#manually added
import numpy as np
import pdb
#manually added
import scipy.stats as stats
import matplotlib.pyplot as plt

class SVParams:
    def __init__(self, phi, sigma, beta):
        self.phi = phi
        self.sigma = sigma
        self.beta = beta

def generator(T, sv_params):
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = np.random.normal(0, sv_params.sigma)
    y[0] = np.random.normal(0, np.sqrt(np.power(sv_params.beta, 2) * np.exp(x[0])))
    for t in range(1, T):
        x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)
        y[t] = np.random.normal(0, np.sqrt(np.power(sv_params.beta, 2) * np.exp(x[t])))
    return x, y

def sis(obs, num_particles, sv_params):
    T = len(obs)
    x = np.zeros([T, num_particles])
    w = np.zeros([T, num_particles])
    log_likelihood = np.zeros(T, dtype='float')

    for t in range(T):
        # YOUR CODE
        """
        We can make SIS function following that description
        when t == 0
        - setting proposal as q = f (prior) -> implicitly, no need to calculate actually as alpha goes to observation itself.
        - setting initial weight as w = alpha, alpha is our weight function, and it is observation when we use prior proposal function
        - normalize weights based on particles
        """
        #Step 1: when t is equal to 0 -> just define x
        if t == 0:
            #get initial distrobution x[t, n] (0~T-1, 0~num_particles-1)
            #this initial distribution refers to assignment description
            #pick one data [t, n] from random (initial sampling)
            x[0] = np.random.normal(0, sv_params.sigma, num_particles)

            #calculate alpha based on adapted proposal
            var = np.exp(x[0]) * np.power(sv_params.beta, 2)
            #get weight update function
            alpha = np.exp(np.power(obs[0], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #first weight is same with alpha
            w[0] = alpha
        else:
            #sampling one step further (calculate f value)
            x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)

            var = np.exp(x[t]) * np.power(sv_params.beta, 2)
            #set our alpha (weight function) as our observation at time t
            alpha = np.exp(np.power(obs[t], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #calculate weight recursively
            #Now,  we are using adopted propopsal,
            weight = w[t-1] * alpha
            w[t] = weight

        #We add weights before normalization
        log_likelihood[t] = np.log(np.sum(w[t])) - np.log(num_particles)
        #normalization
        w[t] = w[t] / np.sum([w[t]])

    log_likelihood_sum = np.sum(log_likelihood)
    return w, x, log_likelihood_sum

def smc_multinomial(obs, num_particles, sv_params):
    T = len(obs)
    x = np.zeros([T, num_particles])
    w = np.zeros([T, num_particles])
    log_likelihood = np.zeros(T, dtype='float')

    for t in range(T):
        #Step 1: when t is equal to 0 -> just define x
        if t == 0:
            #get initial distrobution x[t, n] (0~T-1, 0~num_particles-1)
            #this initial distribution refers to assignment description
            #pick one data [t, n] from random (initial sampling)
            x[0] = np.random.normal(0, sv_params.sigma, num_particles)

            #calculate alpha based on adapted proposal
            var = np.exp(x[0]) * np.power(sv_params.beta, 2)
            #get weight update function
            alpha = np.exp(np.power(obs[0], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #first weight is same with alpha
            w[0] = alpha
        else:
            #sampling one step further (calculate f value)
            x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)

            var = np.exp(x[t]) * np.power(sv_params.beta, 2)
            #set our alpha (weight function) as our observation at time t
            alpha = np.exp(np.power(obs[t], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #calculate weight recursively
            #Now,  we are using adopted propopsal,
            weight = w[t-1] * alpha
            w[t] = weight

        #We add weights before normalization
        log_likelihood[t] = np.log(np.sum(w[t])) - np.log(num_particles)

        #normalization
        w[t] = w[t] / np.sum([w[t]])

        ################## Resampling part ##################
        # We follow seonghwan's lecture note so perform resampling prior to do IS when t > 1
        #Reference: uppsala document http://www.it.uu.se/research/systems_and_control/education/2017/smc/SMC2017.pdf page 90
        #get cumulative weight distributions from previous(t-1) iteration
        phi_dist = np.cumsum(w[t, :])
        #it exceeds 1 sometimes because of floating calculation
        phi_dist[-1] = 1
        #pick uniformly distributed number among its index
        rand_index = np.random.random(len(phi_dist))
        #generate indices for new sampled weights (a^i = F_C^-1(U^i))
        #that is list of selected indices of weights based on multinomial
        u_idx = np.searchsorted(phi_dist, rand_index)
        #replace x and w values to the (role of offsprings inside loop)
        x[t, :] = x[t, :][u_idx]
        w[t, :] = w[t, :][u_idx]
        #####################################################

        #normalization with new weights
        w[t] = w[t] / np.sum([w[t]])

    log_likelihood_sum = np.sum(log_likelihood)
    return w, x, log_likelihood_sum

def smc_stratified(obs, num_particles, sv_params):
    T = len(obs)
    x = np.zeros([T, num_particles])
    w = np.zeros([T, num_particles])
    log_likelihood = np.zeros(T, dtype='float')

    for t in range(T):
        #Step 1: when t is equal to 0 -> just define x
        if t == 0:
            #get initial distrobution x[t, n] (0~T-1, 0~num_particles-1)
            #this initial distribution refers to assignment description
            #pick one data [t, n] from random (initial sampling)
            x[0] = np.random.normal(0, sv_params.sigma, num_particles)

            #calculate alpha based on adapted proposal
            var = np.exp(x[0]) * np.power(sv_params.beta, 2)
            #get weight update function
            alpha = np.exp(np.power(obs[0], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #first weight is same with alpha
            w[0] = alpha
        else:
            #sampling one step further (calculate f value)
            x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)

            var = np.exp(x[t]) * np.power(sv_params.beta, 2)
            #set our alpha (weight function) as our observation at time t
            alpha = np.exp(np.power(obs[t], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #calculate weight recursively
            #Now,  we are using adopted propopsal,
            weight = w[t-1] * alpha
            w[t] = weight

        #We add weights before normalization
        log_likelihood[t] = np.log(np.sum(w[t])) - np.log(num_particles)

        #normalization
        w[t] = w[t] / np.sum([w[t]])

        ################## Resampling part ##################
        pos = (np.random.random(num_particles) + range(num_particles))/num_particles
        u_idx = np.zeros(num_particles, np.int)
        #setting initial position
        pos_idx = 0
        sum_idx = 0
        # We follow Sung's lecture note so perform resampling prior to do IS when t > 1
        #Reference: uppsala document http://www.it.uu.se/research/systems_and_control/education/2017/smc/SMC2017.pdf page 90

        #get cumulative weight distributions from previous(t-1) iteration
        phi_dist = np.cumsum(w[t, :])

        #stratified process
        while pos_idx < num_particles:
            if pos[pos_idx] < phi_dist[sum_idx]:
                u_idx[pos_idx] = sum_idx
                pos_idx = pos_idx + 1
            else:
                sum_idx = sum_idx + 1

        #replace x and w values to the (role of offsprings inside loop)
        x[t, :] = x[t, :][u_idx]
        w[t, :] = w[t, :][u_idx]
        w[t, :] = w[t, :] / np.sum([w[t, :]])
        #####################################################

        #normalization again
        w[t] = w[t] / np.sum([w[t]])

    log_likelihood_sum = np.sum(log_likelihood)
    return w, x, log_likelihood_sum

def main():
	seed = 57832
	np.random.seed(seed)
	T = 100
	params = SVParams(1.0, 0.16, 0.64)
	x, y = generator(T, params)
	print(x[1:5], y[1:5])
	print(x[-5:T], y[-5:T])

	num_particles = 100

	#we modified output to calculate log likelihood (for Task 2.4)
	w_sis, x_sis, ll = sis(y, num_particles, params)
	w_smc, x_smc, ll = smc_multinomial(y, num_particles, params)
	w_str, x_str, ll = smc_stratified(y, num_particles, params)

	# YOUR CODE

	# 1. MSE TEST difffering
	print("Testing with different particles")
	num_particles_list = np.arange(1, 401, 1)

	mse_sis = []
	mse_smc = []
	mse_str = []

	# YOUR CODE
	for num_particles in num_particles_list:
	    w_sis, x_sis, ll = sis(y, num_particles, params)
	    #w_sis, x_sis = filter(y, num_particles, params)
	    #w_smc, x_smc = smc_multinomial(y, num_particles, params)

	    #get point estimate of x: SIS

	    #now we have N number (particle) of estimation for T'th x
	    #get mean square error for T'th x (x[T-1])
	    #test function is applied for each particle  f(x_T^k) =  (x_T^k - x_T^*)^2
	    #and MSE is weighted sum of this test function (according to SLACK group)
	    #in this case we did not perform resampling, so error function is like below
	    error = np.dot(w_sis[-1, :], np.power(x_sis[-1, :] - x[-1], 2))
	    mse_sis.append(error)

	for num_particles in num_particles_list:
	    w_smc, x_smc, ll = smc_multinomial(y, num_particles, params)

	    #get point estimate of x: SIS

	    #now we have N number (particle) of estimation for T'th x
	    #get mean square error for T'th x (x[T-1])
	    #test function is applied for each particle  f(x_T^k) =  (x_T^k - x_T^*)^2
	    #and MSE is weighted sum of this test function (according to SLACK group)
	    error = 1/num_particles*np.sum(np.power(x_smc[-1, :] - x[-1], 2))

	    mse_smc.append(error)
	for num_particles in num_particles_list:
	    w_str, x_str, ll = smc_stratified(y, num_particles, params)

	    #get point estimate of x:

	    #now we have N number (particle) of estimation for T'th x
	    #get mean square error for T'th x (x[T-1])
	    #test function is applied for each particle  f(x_T^k) =  (x_T^k - x_T^*)^2
	    #and MSE is weighted sum of this test function (according to SLACK group)
	    error = 1/num_particles*np.sum(np.power(x_str[-1, :] - x[-1], 2))

	    mse_str.append(error)

	plt.figure(figsize=(16,6))
	plt.plot(range(100),x, label="True X")
	plt.plot(range(100),np.sum(w_sis*x_sis, axis=1), label="SIS")
	plt.plot(range(100),np.sum(w_smc*x_smc, axis=1), label="SMC_MULTINORM")
	plt.plot(range(100),np.sum(w_str*x_str, axis=1), label="SMC_STRATIFIED")
	plt.legend()
	plt.show()

	#2. MSE Trace
	print("Plotting MSE trace")

	#print plot regarding number of particles vs MSE
	plt.figure(figsize=(10,6))
	plt.plot(num_particles_list, mse_sis, 'r--')
	plt.xlabel("Number of particles")
	plt.ylabel("Mean Square Error")
	plt.title("MSE and # of particles on SIS without resampling")
	plt.ylim(0, 3)
	plt.show()

	#print plot regarding number of particles vs MSE
	plt.figure(figsize=(10,6))
	plt.plot(num_particles_list, mse_smc, 'r--')
	plt.xlabel("Number of particles")
	plt.ylabel("Mean Square Error")
	plt.title("MSE and # of particles on SMC with resampling")
	plt.ylim(0, 3)
	plt.show()

	#print plot regarding number of particles vs MSE
	plt.figure(figsize=(10,6))
	plt.plot(num_particles_list, mse_str, 'r--')
	plt.xlabel("Number of particles")
	plt.ylabel("Mean Square Error")
	plt.title("MSE and # of particles on SMC with stratified resampling")
	plt.ylim(0, 3)
	plt.show()

	#3. Historgram
	print("Plotting historgrams")
	plt.hist(w_sis[-1,:], bins=30)
	plt.show()

	plt.hist(w_smc[-1,:], bins=30)
	plt.show()
	
	plt.hist(w_str[-1,:], bins=30)
	plt.show()

if __name__ == '__main__':
	main()
