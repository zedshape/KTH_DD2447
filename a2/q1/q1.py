import numpy as np
import math

def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
    # Data generator. 
    # Input: seed: int, N: int, M: int, K: int, W: int, alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K) 
    # N = # of sequences
    # M = length of sequences
    # K = alphabet
    # W = length of "magic word"
    # alpha_bg = alpha for background
    # alpha_mw = alpha for magic word
    # Output: D: numpy array with shape (N,M), R_truth: numpy array with shape(N), theta_bg: numpy array with shape (K), theta_mw: numpy array with shape (W,K)


    # D is a set of sequences s1, ..., sn generated by the model
    # the start position is not sampled since it's already known


    np.random.seed(seed)        # Set the seed, initializing pseudorandom number generator

    D = np.zeros((N,M))         # Sequence matrix of size NxM
    R_truth = np.zeros(N)       # Start position of magic word of each sequence

    theta_bg = np.zeros(K)      # Categorical distribution parameter of background distribution
    theta_mw = np.zeros((W,K))  # Categorical distribution parameter of magic word distribution

    # YOUR CODE:

     # Generate R_truth
    R_samplespace =  M - W + 1 
    R_truth = np.random.randint(R_samplespace, size = N)      # integers from discrete uniform distribution


    # generate theta
    for i in range(W):
        theta_mw[i] = np.random.dirichlet(alpha_mw)

    theta_bg = np.random.dirichlet(alpha_bg)

    # Generate D 
    for state in range(N):
        for position in range(M):
            #print(" position = " + str(position) + " R_truth = " + str(R_truth[state]))

            if position >= R_truth[state] and position < R_truth[state] + W:
                #print("mw")
                dp = np.nonzero(np.random.multinomial(1,theta_mw[position - R_truth[state]]))[0]
            else:
                #print ("bg")
                dp = np.nonzero(np.random.multinomial(1,theta_bg))[0]

            D[state][position] = dp
   
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

    # s0 initial state
    M = D[0].shape[0]
    B = N*(M-W)

    # marginal likelihood for bg
    alpha_bg_sum = 0
    alpha_mw_sum = 0

    for k in range K:
        alpha_bg_sum += alpha_bg[k]
        alpha_mw_sum += alpha_mw[k]


    #TODO skriv formlerna
    x = math.gamma(5)
    print(x)



 
    

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

if __name__ == '__main__':  #autoruns main
    main()
   