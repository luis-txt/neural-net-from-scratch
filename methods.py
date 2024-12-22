import matplotlib.pyplot as plt
import numpy as np
import random


def sigmoid(x):
    '''
    Calculates sigmoid function for x in R^n

    ---------
    Parameters:
    x : array
        Vector to apply the sigmoid function on.

    -------
    Returns:
    sigmoid of x
    '''
    max = np.log(np.finfo(np.float64).max)
    min = np.log(np.finfo(np.float64).tiny)
    x_clpd = np.clip(x, min, max)
    
    sigmoid = 1 / (1 + np.exp(-x_clpd))
    return sigmoid


def sigmoid_gradient(x):
    '''
    Calculates gradient of sigmoid function for x in R^n

    ---------
    Parameters:
    x : array
        Vector to get the sigmoid gradient on.

    -------
    Returns:
    Gradient of sigmoid x

    '''
    s = sigmoid(x)
    return (s * (1 - s))


def generate_data(M, p_val, p_test, datapath):
    '''
    Generates and seperates data for learning.

    ---------
    Parameters:
    M: Integer
        Total number of datapoints.
    p_val: float
        Percentage of validation data.
    p_test: float
        Percentage of test data.
    datapath: string
        Path to file that holds the data.

    -------
    Returns:
        Array of seperated data-sets
    '''
    print('Generating data...')

    dat = np.loadtxt(datapath,dtype = np.uint8)
    X_data = dat[:,1:]
    Y_data = dat[:,0]
   
    
    # Generate smaller random training, CV and test test set
    M_min = min(M,np.size(X_data, 0))
    M_val   = int(np.floor(M_min * p_val))
    M_test  = int(np.floor(M_min * p_test))
    
    # Pick M_min random indices from the data set
    indices = random.sample(range(np.size(X_data, 0)), M_min)
    
    # Assign indices to the validation/testing/training set
    X_val   = X_data[indices[: M_val]]
    X_test  = X_data[indices[M_val : M_val + M_test]]
    X_train = X_data[indices[M_val + M_test:]]
    
    Y_val   = Y_data[indices[: M_val]]
    Y_test  = Y_data[indices[M_val : M_val + M_test]]
    Y_train = Y_data[indices[M_val + M_test:]]
    
    return [X_val, X_test, X_train, Y_val, Y_test, Y_train]


def visualize_single(X_entry):
    '''
    Visualizes a given character-matrix.

    -------
    Parameters:
    X_entry Array:
        matrix to visualize

    -------
    Returns:
        None.
    '''

    X_quad = np.array(X_entry).reshape(28, 28)

    plt.imshow(X_quad, cmap='gray', vmin=0, vmax=255)
    plt.savefig('visualization_data_single.png')
    plt.show()


def visualize_data(X, Y, P=None):
    '''
    Takes as input 9 images of digits X, the correct labels and optionally the estimated label P.
    
    -------
    Parameters:
    X List: 
        9 datapoints of digits
    Y list: 
        The correct labels
    P list, optional:
        Estimated labels. Defaults to None.
    
    -------
    Returns:
        None.
    '''
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('Visualisierung von 9 Ziffern')
    fig.tight_layout()
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            if index >= len(X):
                break
            title = 'Correc: ' + str(Y[index]) + ' Ges: ' + str(
            P[index]) if P is not None else 'Correc: ' + str(Y[index])
            x_shape = np.array(X[index]).reshape(28, 28)
            axs[i, j].imshow(x_shape, cmap='gray', vmin=0, vmax=255)
            axs[i, j].set_title(title)
    
    plt.savefig('visualization_data_multiple.png')
    plt.show()


def combine_and_sample(A, B, C, sample_size=9):
    '''
    Combines three arrays pairwise using zip, samples a specific number of random triples,
    and unzips them to get three separate arrays.

    -------
    Parameters:
      A array: 
          The first array.
      B array: 
          The second array.
      C array: 
          The third array.
      sample_size integer: 
          The number of random tuples to sample (default is 9).

    -------
    Returns:
        A tuple containing three separate arrays of the sampled elements.
    '''
    combined_AB = list(zip(A, B))
    combined_triples = [(a, b, c) for (a, b) in combined_AB for c in C]

    sampled_triples = random.sample(combined_triples, sample_size)
    sampled_A, sampled_B, sampled_C = zip(*sampled_triples)
    return sampled_A, sampled_B, sampled_C


def print_nn(model):
    '''
    Prints important information of a given neural-network model.

    -------
    Parameters:
    model nn_model:
        Neural network model
    
    -------
    Returns:
        None.
    '''
    print('Number of layers: {model.num_layers}')
    print('Biases:')
    for b in model.biases:
        print(b)
        print('------------------------------------------')
    print('Weights:')
    for w in model.weights:
        print(w)
        print('------------------------------------------')
