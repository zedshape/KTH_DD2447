from __future__ import division
#manually added
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

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

def calculate_posterior_sigma(T, x, a, b, param):
    new_a = a + T/2
    new_b = b+(np.sum([np.power(x[t]-param.phi*x[t-1], 2) for t in range(1, T)]))/2
    return (stats.invgamma.rvs(a=new_a,scale=new_b))

def calculate_posterior_beta(T, x, obs, a, b):
    new_a = a + T/2
    new_b = b+(np.sum([np.exp(-x[t])*np.power(obs[t], 2) for t in range(T)]))/2
    return (stats.invgamma.rvs(a=new_a, scale=new_b))

def conditional_smc(obs, num_particles, sv_params, k):
    #k is the index that smc must hold on!
    T = len(obs)
    x = np.zeros([T, num_particles])
    xk = np.zeros([T])
    w = np.zeros([T, num_particles])

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
            xk[0] = x[0, k]
        else:
            #sampling one step further (calculate f value)
            x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)

            var = np.exp(x[t]) * np.power(sv_params.beta, 2)
            #set our alpha (weight function) as our observation at time t
            alpha = np.exp(np.power(obs[t], 2) / (-2*var)) / np.sqrt(2*np.pi*var)

            #calculate weight recursively
            #Now, as we are using prior propopsal, our weight is just multiplying our observation again and again
            weight = w[t-1] * alpha
            w[t] = weight
            #after resampling, tractable k'th element is always located in the end
            xk[t] = x[t, -1]

        #normalization
        w[t] = w[t] / np.sum([w[t]])

        ################## Resampling part ##################
        phi_dist = np.cumsum(w[t, :])
        phi_dist[-1] = 1
        rand_index = np.random.random(len(phi_dist))
        u_idx = np.searchsorted(phi_dist, rand_index)
        #start to track k always locating last part
        if t == 0:
            u_idx = np.append(np.random.choice(u_idx, len(u_idx)-1), k)
        else:
            u_idx = np.append(np.random.choice(u_idx, len(u_idx)-1), T-1) #k is located at last

        x[t, :] = x[t, :][u_idx]
        w[t, :] = w[t, :][u_idx]
        #####################################################

        #normalization with new weights
        w[t] = w[t] / np.sum([w[t]])

    return w, xk

def pg(obs, num_particles=100, num_mcmc_iter=10000):
    burn_in_rate = 0.8
    num_burn_in = int(burn_in_rate * num_mcmc_iter)

    #distribution parameters
    a = 0.01
    b = 0.01
    phi = 1.0 #fixed
    T = len(obs)

    #return values
    X = np.zeros([num_mcmc_iter, T])
    params = [] # list of SV_params

    #Gibbs sampling
    for i in range(num_mcmc_iter + num_burn_in):
        if i == 0:
            #inverse gamma prior
            sigma = stats.invgamma.rvs(0.01, 0.01)
        else:
            #inverse gamma posterior
            sigma = np.sqrt(calculate_posterior_sigma(T, x, a, b, param))
        if i == 0:
            beta = stats.invgamma.rvs(0.01, 0.01)
        else:
            beta = np.sqrt(calculate_posterior_beta(T, x, obs, a, b))
        if i < 10:
            print(sigma, beta)

        #print(sigma, beta)
        #parameter update
        param = SVParams(phi, sigma, beta)
        params.append(param)

        #update x first with sigma to prevent infinity
        k = int(np.random.choice(range(num_particles), 1))
        w, x = conditional_smc(obs, num_particles, param, k)

        #set return value after burn-in
        if i >= num_burn_in:
            X[i-num_burn_in] = x

    #After looping, discard burn in parameters
    params = params[num_burn_in:]

    return X, params

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
    V_hat = (1-(1/n))*W + 1/n*B

    #R
    R_hat = np.sqrt(V_hat/W)

    return R_hat

def main():
	seed = 12345 #CAREFUL: IT CAN MAKE CODE BREAK DOWN
	np.random.seed(seed)

	#hyperparameters
	sigma = 0.16
	beta = 0.64
	phi = 1.0
	T = 100

	#ground truth (observation)
	param = SVParams(phi, sigma, beta)
	_, obs = generator(T, param)

	print("It will return some warning at the first, because of inverse gamma's instability")
	print("Depending on seed, it will not reduce from infinity then 'Domain error' will be throwed")
	print("It is because of the property of inverse gamma during getting posterior, so please carefully choose seed")
	#run pg with observation
	X_pg, params = pg(obs)

	sigmas = []
	betas = []
	for i in params:
	    sigmas.append(i.sigma)
	    betas.append(i.beta)

	#####plotting area
    #We commented in all plt.show() for submission
    #Grader can comment them out to check our result
	print("Plotting sigma and beta squared distribution")
	sigmas = np.array(sigmas)
	betas = np.array(betas)
	sigmas_squared = sigmas**2
	beta_squared = betas**2

	fig, axes = plt.subplots(1,2, figsize=(16, 6))
	axes[0].hist(sigmas_squared[sigmas_squared<2], bins=100)
	axes[0].axvline(x=0.16**2, color='r')
	axes[1].hist(beta_squared[beta_squared<2], bins=100)
	axes[1].axvline(x=0.64**2, color='r')
	#plt.show()

	#check for convergence
	# it takes a lot of time so that we comment them in
	"""
	multiple_gibbs = [pg(obs) for _ in range(3)]

	multiple_sigmas = [[j.sigma for j in i[1]] for i in multiple_gibbs]
	multiple_sigmas = np.array(multiple_sigmas)

	multiple_betas = [[j.beta for j in i[1]] for i in multiple_gibbs ]
	multiple_betas = np.array(multiple_betas)

	diagnosis_results = np.array([diagnosis(multiple_sigmas[:,0:t]) for t in range(2,10000)])
	plt.figure(figsize=(8,4))
	plt.title('sigma: Diagnosis result(Convergence rate) using Gelman-Rubin diagnostic')
	plt.plot(diagnosis_results)
	plt.show()

	diagnosis_results = np.array([ diagnosis(multiple_betas[:,0:t]) for t in range(2,10000)])
	plt.figure(figsize=(8,4))
	plt.title('beta: Diagnosis result(Convergence rate) using Gelman-Rubin diagnostic')
	plt.plot(diagnosis_results)
	plt.show()
	"""

if __name__ == '__main__':
	main()
