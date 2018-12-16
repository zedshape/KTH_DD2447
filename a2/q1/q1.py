import numpy as np

def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
    # Data generator. 
    # Input: seed: int, N: int, M: int, K: int, W: int, alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K) 
    # Output: D: numpy array with shape (N,M), R_truth: numpy array with shape(N), theta_bg: numpy array with shape (K), theta_mw: numpy array with shape (W,K)

    np.random.seed(seed)        # Set the seed

    D = np.zeros((N,M))         # Sequence matrix of size NxM
    R_truth = np.zeros(N)       # Start position of magic word of each sequence

    theta_bg = np.zeros(K)      # Categorical distribution parameter of background distribution
    theta_mw = np.zeros((W,K))  # Categorical distribution parameter of magic word distribution

    # YOUR CODE:
    # Generate D, R_truth, theta_bg, theta_mw. Please use the specified data types and dimensions. 

    return D, R_truth, theta_bg, theta_mw

def gibbs(D, alpha_bg, alpha_mw, num_iter, W):
    # Gibbs sampler. 
    # Input: D: numpy array with shape (N,M),  alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K), num_iter: int, W: int
    # Output: R: numpy array with shape(num_iter, N)
    
    N = D.shape[0]
    R = np.zeros((num_iter, N)) # Store samples for start positions of magic word of each sequence

    # YOUR CODE:
    # Implement gibbs sampler for start positions. 

    for n in range(num_iter):
        pass

    return R

def main():
    seed = 123

    N = 20
    M = 10
    K = 4
    W = 5
    alpha_bg = np.ones(K)
    alpha_mw = np.ones(K) * 0.9

    num_iter = 1000
    
    print("Parameters: ", seed, N, M, K, W, num_iter)
    print(alpha_bg)
    print(alpha_mw)
    
    # Generate synthetic data.
    D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw)
    print("\nSequences: ")
    print(D)
    print("\nStart positions (truth): ")
    print(R_truth)

    # Use D, alpha_bg and alpha_mw to infer the start positions of magic words. 
    R = gibbs(D, alpha_bg, alpha_mw, num_iter, W)
    print("\nStart positions (sampled): ")
    print(R[0,:])
    print(R[1,:])

    # YOUR CODE:
    # Analyze the results. Check for the convergence. 

if __name__ == '__main__':
    main()
   