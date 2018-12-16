import numpy as np

class HyperParams:
	def __init__(self, alpha, a_0, a_1, d_0):
		self.alpha = alpha
		self.a_0 = a_0
		self.a_1 = a_1
		self.d_0 = d_0

def generator(nobs, hyper_params):
	Z = np.zeros(nobs)
	b = np.zeros(nobs)
	d = np.zeros(nobs)
	phi = np.zeros(nobs)
	# YOUR CODE goes here
	return b, d, phi, Z

def collapsed_gibbs(b, d, hyper_params, num_iter = 2000):
	nobs = len(b)
	Z = np.zeros(nobs)
	phi = np.zeros(nobs)
	for t in range(num_iter):
		# YOUR CODE GOES HERE
		pass
	return phi, Z

def main():
	seed = 57832
	np.random.seed(seed)
	nobs = 10
	params = HyperParams(1, 1, 1, 1000)
	b, d, phi, Z = generator(nobs, params)
	print(b, d)	

	# YOUR CODE

if __name__ == '__main__':
	main()
