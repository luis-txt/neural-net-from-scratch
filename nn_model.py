import numpy as np
import random as rnd
import matplotlib.pyplot as plt

from methods import generate_data, sigmoid, sigmoid_gradient

class nn_model:
    '''
    Neural network model

    ----------
    Attributes:
    sizes (list): 
        A list of integers representing the number of neurons in each
        layer of the neural network.
    X_val (list): 
        The validation input data.
    X_test (list): 
        The test input data.
    X_train (list): 
        The training input data.
    Y_val (list): 
        The validation output data.
    Y_test (list): 
        The test output data.
    Y_train (list): 
        The training output data.
    num_layers (int): 
        The number of layers in the neural network.
    biases (list):
        A list of bias vectors for each layer of the neural network.
    weights (list): 
        A list of weight matrices for each layer of the neural network.
    progress (list): 
        A placeholder for tracking the training progress.
    costs (list): 
        Training costs.
    '''

    def __init__(self, M, p_val, p_test, sizes, dataname):
        '''
        Initialize the NN-Model class.

        ----------
        Parameters:
        sizes (list): 
            A list of integers representing the number of neurons in each 
            layer of the network.
        M (int):
            The number of training examples.
        p_val (float):
            The proportion of examples to be used for validation.
        p_test (float): 
            The proportion of examples to be used for testing.
        dataname (string): 
            The name of the dataset.

        ----------
        Returns:
        None.
        '''
        self.sizes = sizes
        self.M = M
        self.progress = []
        self.costs = []

        if dataname != False:
            (self.X_val, self.X_test, self.X_train, self.Y_val, self.Y_test,
            self.Y_train) = generate_data(M, p_val, p_test, dataname)

            self.Y_val = self.nn_one_hot(self.Y_val)
            self.Y_test = self.nn_one_hot(self.Y_test)
            self.Y_train = self.nn_one_hot(self.Y_train) 
            
             # Normalization
            mu, sigma, indices = self.getMeanAndVariance(self.X_train)
            self.X_train_norm = self.transform_data(self.X_train, mu, sigma, indices) 
            self.X_val_norm = self.transform_data(self.X_val, mu, sigma, indices)
            self.X_test_norm = self.transform_data(self.X_test, mu, sigma, indices)

            # Update sizes to match normalized training data
            self.sizes[0] = self.X_train_norm.shape[1]

        self.nn_build(self.sizes)


    def nn_build(self, sizes, eps=0.12):
        '''
        Build the neural network.

        ----------
        Parameters:
        Sizes (list): 
            A list of integers representing the number of neurons in each layer 
            of the neural network.
        Eps (float): 
            The epsilon value for initializing the weights.

        ----------
        Returns:
            None.
        '''
        print('Building neural network...')
        self.sizes = sizes
        self.num_layers = len(sizes)
       
        # Fill biases with 0 and weights with random values with respect to eps
        self.biases = [np.zeros(size) for size in sizes[1:]]
        self.weights = [np.random.randn(sizes[i - 1], sizes[i]) * eps for i in range(1, self.num_layers)]


    def nn_one_hot(self, Y):
        '''
        Builds neural network with layer sizes given in sizes. The first layer is
        the input layer and the last layer the output layer. The parameter eps 
        the standard deviation for
        the random (Gaussian with 0 mean) weight initialization. 
        The biases are initialized to zero.

        ----------
        Parameters:
        sizes : list
            Layer sizes.
        eps : float
            Standard deviation for weight initialization.

        ----------
        Returns:
        - One-Hot matrix of input vector.
        '''
        Y = np.array(Y)
        one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Create matrix
        one_hot_Y[np.arange(Y.size), Y] = 1 # Fill matrix
        return one_hot_Y


    def nn_one_hot_inv(self, one_hot_Y):
        '''
        Transformes Y into a one-hot version
        
        ----------
        Parameters:
        one_hot_Y : array
            One-Hot encoded vector.

        ----------
        Returns:
        - Original vector of the input one-hot matrix.
        '''
        return np.argmax(one_hot_Y, axis=1)


    def nn_forward(self,X):
        '''
        Forward propagation through neural network given by nn_model for data X.

        ----------
        Parameters:
        X : array
            Data X.

        ----------
        Returns:
        A : list of arrays
            Activations.
        Z : list of arrays
            Weighted inputs.

        '''
        A = [X]
        Z = [np.dot(A[0], self.weights[0]) + self.biases[0]]
       
        for l in range(1, self.num_layers - 1):
            W_l = self.weights[l]
            B_l = self.biases[l]
            A_l = sigmoid(Z[l-1])
            Z_l = np.dot(A_l, W_l) + B_l
            Z.append(Z_l)
            A.append(A_l)

        A.append(sigmoid(Z[-1]))

        return A, Z


    def nn_cost(self, Y, A_out):
        '''
        Calculate cost function for neural network. Uses output from nn_forward.
        This function is currently only neccesary for nn_check.
        
        ----------
        Parameters:
        Y : array
            True output.    
        A_out : TYPE
            Last layer of A from nn_forwards output.

        -------
        Returns:
        cost : float
            Evaluation of cost function.

        '''
        # Vectorized formula from "Künstliche Neuronale Netze"
        cost = (Y * np.log(A_out) + (1 - Y) * np.log(1 - A_out)).sum()
        return -(cost / len(Y))


    def getMeanAndVariance(self,X):
        '''
        Prepares normalization.
        Returns the parameters necessary to apply the transformation for normalization.
        
        ----------
        Parameters:
        X : array
            Data X.

        -------
        Returns:
        mu : array
            Feature mean with removed 0 variance features.
        sigma : array
            Standard deviation per feature with removed 0 variance features.
        indices : array
            Index array of features with nonzero variance.
        '''
        mu = np.mean(X, axis=0)
        sigma = np.var(X,axis=0)
        indices = np.nonzero(sigma != 0)[0] # Get indices of non-zero variance features

        return mu, sigma, indices 


    def transform_data(self, X, mu, sigma, indices):
        '''
        Transform non-normalized data X into normalized data using output of
        normalize_data. This is necessary to get training, validation and test data 
        into the correct format.

        ----------
        Parameters:
        mu : array
            Feature mean with removed 0 variance features.
        X  : array (2d array)
            Data matrix to be normalised
        sigma : array
            Standard deviation per feature with removed 0 variance features.
        indices : array
            Index array of features with nonzero variance.
            
        -------
        Returns:
        X_norm : array
            Column-wise normalized version of X with removed 
            0 variance features.

        ''' 
        sigma = sigma[indices]
        mu = mu[indices]

        X_norm = (X[:, indices] - mu) / sigma
        return X_norm


    def nn_backward(self, X, Y, A, Z):
        '''
        Backward propagation through neural network given by nn_model for data X, Y.
        Output from nn_forward is used.

        ----------
        Parameters:
        X : array
            Data X.
        Y : array
            Data Y.
        A : list of arrays
            Activations.
        Z : list of arrays
            Weighted inputs.

        -------
        Returns:
        W_gradients : TYPE
            DESCRIPTION.
        b_gradients : TYPE
            DESCRIPTION.
        '''
        W = self.weights

        # Compute deltas per layer
        delta = [A[-1] - Y]
        for l in range(self.num_layers - 2, 0, -1):
            delta.append(delta[-1].dot(W[l].T) * sigmoid_gradient(Z[l - 1]))

        delta.reverse()

        # Calculate gradients
        W_gradients = [1 / X.shape[0] * A[l].T.dot(delta[l]) for l in range(self.num_layers - 1)]
        b_gradients = [1 / X.shape[0] * np.sum(delta[l], axis=0) for l in range(self.num_layers - 1)]

        return W_gradients, b_gradients


    def nn_train(self, batch_size, maxepoch, K, eta_0, eta_K):
        '''
        Stochastic gradient descent with replacement and update rule of learning rate
        from equation 8.14 in "Deep Learning".
        We use the old success rate and cost at current mini-batch for visualization of progress.
        This can be cheaply evaluated.

        ----------
        Parameters
        batch_size : integer
            Used mini-batch size.
        maxepoch : integer
            Number of epochs for training.
        K : integer
            Learning rate parameter.
        eta_0 : float
            Learning rate parameter.
        eta_K : float
            Learning rate parameter.
        
        -------
        Returns:
        None
        '''
        k = 0
        for epoch in range(maxepoch):
            # Shuffle the training data
            train_data = list(zip(self.X_train_norm, self.Y_train))
            rnd.shuffle(train_data)
            self.X_train_norm, self.Y_train = zip(*train_data)
          
            # Partition into batches
            batches = [
                (self.X_train_norm[i:i + batch_size], self.Y_train[i:i + batch_size])
                for i in range(0, len(self.X_train_norm), batch_size)
            ]

            for X_batch, Y_batch in batches:
                X_batch = np.array(X_batch)
                Y_batch = np.array(Y_batch)

                A, Z = self.nn_forward(X_batch)
                W_grad, b_grad = self.nn_backward(X_batch, Y_batch, A, Z)
               
                # Calculate step-size
                if k <= K:
                    eta_k = (1 - (k/K))*eta_0 + (k/K)*eta_K
                else:
                    eta_k = eta_K

                # Update biases and weights
                for l in range(len(self.weights)):
                    self.weights[l] -= (eta_k / batch_size) * W_grad[l]
                    self.biases[l] -= (eta_k / batch_size) * b_grad[l]

                k += 1

            A, _ = self.nn_forward(self.X_val_norm)
            A_out = A[-1]
            cost = self.nn_cost(self.Y_val, A_out)
            sccss = self.nn_successrate(self.Y_val, A_out)
            self.costs.append(cost)
            self.progress.append(sccss)
            print(f'Epoch: {epoch+1}, Success Rate: {sccss:.4f}, Cost: {cost:.4f}')


    def nn_successrate(self,Y, A_out):
        '''
        Calculates prediction and success rate of neural network given data Y and
        results from nn_forward.

        ----------
        Parameters:
        Y : array
            True output.
        A_out : TYPE
            Last layer of A from nn_forwards output.

        -------
        Returns:
        rate : float
            Success rate.

        '''
        Y_pred = np.argmax(A_out, axis=1)
        Y_true = np.argmax(Y, axis=1)
        success_rate = np.mean(Y_pred == Y_true)
        return success_rate


    def visualize_cost_progress(self):
        '''
	    Visualize the tranining progress.

        ----------
        Parameters:
        None

        -------
        Returns:
        None
        '''
        plt.plot(self.costs, label='Cost')
        plt.plot(self.progress, label='Validation Success Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Progress')
        plt.legend()
        plt.show()
