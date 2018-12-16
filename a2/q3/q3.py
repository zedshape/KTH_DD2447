import numpy as np
import pdb

class SVParams:
	def __init__(self, phi, sigma, beta):
		self.phi = phi
		self.sigma = sigma
		self.beta = beta

def generator(T, sv_params):
	x = np.zeros(T)
	y = np.zeros(T)
	x[0] = np.random.normal(0, np.power(sv_params.sigma, 2))
	y[0] = np.random.normal(0, np.power(sv_params.beta, 2) * np.exp(x[0]))
	for t in range(1, T):
		x[t] = np.random.normal(sv_params.phi * x[t-1], np.power(sv_params.sigma, 2))
		y[t] = np.random.normal(0, np.power(sv_params.beta, 2) * np.exp(x[t]))
	return x, y

def sis(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			# YOUR CODE
			pass
	return w, x

def smc_multinomial(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			# YOUR CODE
			pass
	return w, x

def smc_stratified(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			# YOUR CODE
			pass
	return w, x

def main():
	seed = 57832
	np.random.seed(seed)
	T = 500
	params = SVParams(0.6, 0.7, 0.9)
	x, y = generator(T, params)
	print(x[1:5], y[1:5])
	print(x[-5:T], y[-5:T])
	
	num_particles = 100
	sis(y, num_particles, params)
	smc_multinomial(y, num_particles, params)
	smc_stratified(y, num_particles, params)

	# YOUR CODE

if __name__ == '__main__':
	main()