import numpy as np
import math
import matplotlib.pyplot as plt

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
    print("len R " + str(len(R)))

    # YOUR CODE:
    # Implement gibbs sampler for start positions. 
  
    M = D[0].shape[0]
    B = N*(M-W)
    K = alpha_mw.shape[0]

    # randomly setting up initial state r0
    R0 = np.random.randint((M - W + 1), size = N)
    R02 = np.random.randint((M - W + 1), size = N)

    samples = []
    samples.append(R02)      #starting sequence

   # plot = plt.figure()

    for n in range(num_iter):  
        positions = []        # this will be a row in R       
        current_state=samples[-1]   # the last sampled sequence

        for s in range(N):
            pos_proba = []
        
            for r_i in range(M - W + 1):         # loop over possible start positions
                count_mw = np.ones((W,K))         # counting occurances of each element at each position in mw
                count_bg = np.ones(K)             # counting occurances of elements in bg

                seq_bg = D
                seq_mw = np.zeros((N,W))
                for a in range(W):
                    seq_mw[:,a] = D[:,r_i+a]
                    seq_bg = np.delete(seq_bg,r_i,axis=1)
                seq_bg = np.delete(seq_bg,s,axis=0)
                seq_mw = np.delete(seq_mw,s,axis=0)
                seq_mw = np.delete(seq_mw,1,axis=1)
                    
                for c in range(N-1):
                    for m in range(M-W):
                    #counts character occurance for char in bg
                        count_bg[seq_bg[c][m]] += 1
                    
                    #counts character occurance for every char in every position of mw
                    for w in range(W-1):
                        count_mw[w][seq_mw[c][w]] +=1

                C = (math.gamma(np.sum(alpha_bg)) / math.gamma(B + np.sum(alpha_bg)))

                class_probs = [ math.gamma(count_bg[k] + alpha_bg[k]) / math.gamma(alpha_bg[k])  for k in range(K)]
        
                p_bg = C * np.prod(class_probs) 

                C2 = math.gamma(np.sum(alpha_mw))/math.gamma(N * W + np.sum(alpha_mw))

                class_probs_j = []
                for j in range(W):
                    class_probs_jk = [ math.gamma(count_mw[j][k] + alpha_mw[k]) / math.gamma(alpha_mw[k])  for k in range(K) ]
                    class_probs_jk = C2 * np.prod(class_probs_jk) 
                    class_probs_j.append(class_probs_jk)

                p_mw = class_probs_j
                print(class_probs_j)
                p =  p_bg + np.prod(p_mw) 
                
                pos_proba.append(p)

            #normalize
            p = np.asarray(pos_proba)
            #p = np.exp(p - np.max(p))
            p = p / np.sum(p)


            multi_samp = np.random.multinomial(1,p)
            position = np.argmax(multi_samp)        
            current_state[s] = position
            positions.append(position)

            samples.append(np.array(positions))

        R[n]= np.array(positions)

    return R


    #plot
    

def main():
    seed = 123

    N = 20
    M = 10
    K = 4
    W = 5
    alpha_bg = np.ones(K)
    alpha_mw = np.ones(K) * 0.9

    num_iter = 1000 #CHANGE BACK TO 1000
    
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
    plt.title("TEST")
    step = 1
    x_axis = np.zeros(num_iter/step)
    y_axis = np.zeros(num_iter/step)
    #print(x_axis.shape)
    #print(y_axis.shape)

    # plot, axes = plt.subplots(6,1)
    plt.xlabel('Iteration')
    plt.ylabel('R[n]')


    pos = 6
    i =0
    for n in range(num_iter):
        #if n % (step) == 0:
        #if n >= 900:
            #print(i)
            x_axis[i] = n
            y_axis[i] = R[n][pos]
            #print(R[n])
            i +=1

    #  print(x_axis)
    #  print(y_axis)
    t = np.ones(K) * R_truth[pos]
    plt.grid(True)
    plt.axis([x_axis[0], i*step, 0, M])
    plt.plot(x_axis, y_axis, color = "b" )
    #plt.plot(x_axis , t, color ="r")
    plt.show()
    # Analyze the results. Check for the convergence. 

if __name__ == '__main__':  #autoruns main
    main()
   