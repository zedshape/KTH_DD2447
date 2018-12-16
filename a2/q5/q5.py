
def pg(obs, num_particles=100, num_mcmc_iter=2000):
	T = len(obs)
	X = np.zeros([num_mcmc_iter, T])
	params = [] # list of SV_params
	# YOUR CODE
	return X, params
