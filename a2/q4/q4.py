# import and use functions from q3.py
from q3 import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pdb

def calculate_posterior_sigma(T, x, a, b, param):
    new_a = a + T/2
    new_b = b+(np.sum([np.power(x[t]-param.phi*x[t-1], 2) for t in range(1, T)]))/2
    return (stats.invgamma.rvs(a=new_a,scale=new_b))

def calculate_posterior_beta(T, x, obs, a, b):
    new_a = a + T/2
    new_b = b+(np.sum([np.exp(-x[t])*np.power(obs[t], 2) for t in range(T)]))/2
    return (stats.invgamma.rvs(a=new_a, scale=new_b))

def smc_pmmh(obs, num_particles, sv_params):
    T = len(obs)
    x = np.zeros([T, num_particles])
    w = np.zeros([T, num_particles])
    likelihood = np.zeros(T, dtype='float')

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
            #Now, as we are using prior propopsal, our weight is just multiplying our observation again and again
            weight = w[t-1] * alpha
            w[t] = weight

        #We add weights before normalization
        likelihood[t] = np.sum(w[t])/num_particles

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

    likelihood_total = np.product(likelihood)
    return w, x, likelihood_total

def pmmh(obs, num_particles=100, num_mcmc_iter=2000):
    #distribution parameters
    T = len(obs)
    a = 0.01
    b = 0.01

    #return values
    X = np.zeros([num_mcmc_iter, T])
    params = [] # list of SV_params

    for it in range(num_mcmc_iter):
        if it == 0:

            #initialization step
            #set initial parameter values using uniform distribution (0, 2)
            sig_int = np.random.uniform(0, 2)
            beta_int = np.random.uniform(0, 2)
            #sig_int = stats.invgamma.rvs(0.01, 0.01)
            #beta_int = stats.invgamma.rvs(0.01, 0.01)

            #make parameter object (phi is 1.0)
            params.append(SVParams(1.0, sig_int, beta_int))

            ###X sample selection based on weight (multinomial sampling)
            w_tmp, x_tmp, old_prod = smc_pmmh(obs, num_particles, params[0])

            #MULTINOMIAL RESAMPLING IS USED TO PICK ONE x VECTOR
            phi_dist = np.cumsum(w_tmp[T-1, :])
            phi_dist[-1] = 1
            rand_index = np.random.random(1)
            u_idx = np.searchsorted(phi_dist, rand_index)
            ####################################################

            X[it, :] = x_tmp[:, u_idx].reshape(T)

        else:
            w_old = w_tmp
            x_old = x_tmp

            #random walk proposal
            sigma_tmp = np.abs(params[it-1].sigma + np.random.normal(scale=0.1))
            beta_tmp = np.abs(params[it-1].beta + np.random.normal(scale=0.1))
            #sigma_tmp = np.sqrt(calculate_posterior_sigma(T, X[it-1], a, b, params[it-1]))
            #beta_tmp = np.sqrt(calculate_posterior_beta(T, X[it-1], obs, a, b))

            #sigma_tmp = np.sqrt(calculate_posterior_sigma(T, X[it-1], a, b, params[it-1]))
            #beta_tmp = np.sqrt(calculate_posterior_beta(T, X[it-1], obs, a, b,  params[it-1]))
            #print(sigma_tmp, beta_tmp)
            #for new sigma and beta
            #candidateParams = SVParams(1.0, sigma_tmp, beta_tmp)
            candidateParams = SVParams(1.0, sigma_tmp, beta_tmp)
            w_tmp, x_tmp, new_prod = smc_pmmh(obs, num_particles, candidateParams)

            #compare probs (Z * P / Z * P)
            alpha = min(1, new_prod/old_prod)
            #with alpha probability, change setting
            if np.random.choice(2, p=[alpha, 1-alpha]) == 0:
                new_sigma = sigma_tmp
                new_beta = beta_tmp
                old_prod = new_prod
            else:
                new_sigma = params[it-1].sigma
                new_beta = params[it-1].beta
                x_tmp = x_old
                w_tmp = w_old

            #get X value using smc and new beta and sigma
            candidateParams = SVParams(1.0, new_sigma, new_beta)
            params.append(candidateParams)

            #MULTINOMIAL RESAMPLING IS USED TO PICK ONE x VECTOR
            phi_dist = np.cumsum(w_tmp[-1, :])
            phi_dist[-1] = 1
            rand_index = np.random.random(1)
            u_idx = np.searchsorted(phi_dist, rand_index)
            ####################################################

            X[it, :] = x_tmp[:, u_idx].reshape(T)

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
    ###Below is part of SMC in Task 2.4
    #Test differing sigma and beta
    print("Test differing sigma and beta")

    #####plotting area
    #We commented in all plt.show() for submission
    #Grader can comment them out to check our result

    #set initial N value
    num_particles = 100
    T = 100

    # Generate samples using beta=0.64, sigma=0.16
    params_groundtruth = SVParams(1.0, 0.16, 0.64)
    x_truth, y_truth = generator(T, params_groundtruth)


    #make a reasonable 8 by 8 coarse grid
    #0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2
    beta_list = np.arange(0.25, 2.1, 0.25)
    sigma_list = np.arange(0.25, 2.1, 0.25)

    #ground truth is 0.16 * 0.64

    #set initial dictionary
    likelihoods = {}

    #generate sv_params for 8 by 8 grid
    for sigma in sigma_list:
        for beta in beta_list:
            likelihoods[(sigma, beta)] = []
            #run SMC 10 times
            for _ in range(10):
                #different random seed
                np.random.seed(_)

                #set new params
                params = SVParams(1.0, sigma, beta)
                #get resampling with current alpha and beta
                w_smc_ll, x_smc_ll, ll = smc_multinomial(y_truth, num_particles, params)
                likelihoods[(sigma, beta)].append(ll)
    keys = list(likelihoods.keys())
    plt.figure(figsize=(20,6))
    plt.boxplot(likelihoods.values())
    plt.xticks(range(64), keys, rotation=45, rotation_mode="anchor")
    plt.xlabel("composition of alpha and beta")
    plt.ylabel("log likelihood")
    #plt.show()

    #Test differing T and N
    print("Test differing T and N ")
    print("Differing N with set T")
    particles = np.arange(100, 1001, 100)

    likelihoods = {}

    for num_particles in particles:
        likelihoods[num_particles] = []
        #run SMC 10 times
        for _ in range(10):
            #different random seed
            np.random.seed(_)
            __, __, ll = smc_multinomial(y_truth, num_particles, params_groundtruth)
            likelihoods[num_particles].append(ll)


    keys = list(likelihoods.keys())
    plt.figure(figsize=(5,7))
    plt.boxplot(likelihoods.values())
    plt.xticks(range(10), keys, rotation=45, rotation_mode="anchor")
    plt.xlabel("Number of particles")
    plt.ylabel("log likelihood")
    #plt.show()

    print("Differing T with set N")
    num_particle = 100

    Ts = np.arange(100, 1001, 100)
    likelihoods = {}

    params_groundtruth = SVParams(1.0, 0.16, 0.64)

    for tmp_T in Ts:
        likelihoods[tmp_T] = []
        #run SMC 10 times
        for _ in range(10):
            #different random seed
            np.random.seed(_)
            x_sample, y_sample = generator(tmp_T, params_groundtruth)
            _, _, ll = smc_multinomial(y_sample, num_particle, params_groundtruth)
            likelihoods[tmp_T].append(ll)

    """
    plt.figure(figsize=(16,6))
    plt.plot(Ts, likelihoods)
    """
    keys = list(likelihoods.keys())
    plt.figure(figsize=(5,7))
    plt.boxplot(likelihoods.values())
    plt.xticks(range(10), keys, rotation=45, rotation_mode="anchor")
    plt.xlabel("Number of timeslots (T)")
    plt.ylabel("log likelihood")
    #plt.show()

    ############### PMMH
    print("Start PMMH Test")

    seed = 65431
    np.random.seed(seed)

    #hyperparameters
    sigma = 0.16
    beta = 0.64
    phi = 1.0
    T = 100

    #ground truth
    params = SVParams(1.0, 0.16, 0.64)
    _, obs = generator(T, params)

    #run pmmh with observation
    X_pmmh, pmmhresult = pmmh(obs)

	#####plotting area
    #We commented in all plt.show() for submission
    #Grader can comment them out to check our result

    sigmas = []
    betas = []

    for i in pmmhresult:
        sigmas.append(i.sigma)
        betas.append(i.beta)

    #sigma and beta fluctuation
    print("Plotting sigma and beta fluctuation and histogram")
    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes[0].plot(sigmas[1:])
    axes[0].axhline(y=0.16, color='r')
    axes[1].hist(sigmas[1:], bins=50)
    axes[1].axvline(x=0.16, color='r')
    #plt.show()

    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes[0].plot(betas)
    axes[0].axhline(y=0.64, color='r')
    axes[1].hist(betas, bins=50)
    axes[1].axvline(x=0.64, color='r')
    #plt.show()

    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    sigmas_squared = np.array(sigmas)
    sigmas_squared = sigmas_squared**2
    axes[0].hist(sigmas_squared, bins=100)
    axes[0].axvline(x=0.16**2, color='r')

    betas_squared = np.array(betas)
    betas_squared = betas_squared**2
    axes[1].hist(betas_squared, bins=100)
    axes[1].axvline(x=0.64**2, color='r')
    #plt.show()

    #check for convergence
	# it takes a lot of time so that we comment them in

    """

    print("Checking convergence")
    multiple_pmmhs = [pmmh(obs) for _ in range(5)]
    multiple_sigmas = [[j.sigma for j in i[1]] for i in multiple_pmmhs]
    multiple_sigmas = np.array(multiple_sigmas)

    multiple_betas = [[j.beta for j in i[1]] for i in multiple_pmmhs]
    multiple_betas = np.array(multiple_betas)

    diagnosis_results = np.array([diagnosis(multiple_sigmas[:,0:t]) for t in range(2,2000)])
    plt.figure(figsize=(8,4))
    plt.title('sigma: Diagnosis result(Convergence rate) using Gelman-Rubin diagnostic')
    plt.plot(diagnosis_results)
    #plt.show()

    diagnosis_results = np.array([diagnosis(multiple_betas[:,0:t]) for t in range(2,2000)])
    plt.figure(figsize=(8,4))
    plt.title('beta: Diagnosis result(Convergence rate) using Gelman-Rubin diagnostic')
    plt.plot(diagnosis_results)
    #plt.show()
    """


if __name__ == '__main__':
	main()
