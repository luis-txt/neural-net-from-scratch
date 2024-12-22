import os

from methods import *
from nn_model import nn_model

def test_nn_build_and_one_hot(file):
    '''
    Test the nn_build function of the Network and the one-hot encoding.
    It then visualizes the data.

    Parameters:
        file (str): The file path.

    Returns:
        None
    '''
    print('Testing nn_build of the Network...')
    model = nn_model(10000, 0.1, 0.1, [784, 20, 10], file)
    print('Model with 1 hidden layer of 20 neurons:')
    print_nn(model)

    print('Testing one-hot and its inverse...')
    original_Y = model.nn_one_hot_inv(model.Y_val)
    visualize_data(model.X_val, original_Y)


def test_generate_data(path):
    '''
    Test the data generation and visualize the data.

    Args:
        path (str): The path to the data.

    Returns:
        None
    '''
    print('Testing data generation and visualisation...')
    X_val, _, X_train, Y_val, _, _ = generate_data(11, 0.9, 0.01, datapath=path)

    visualize_single(X_train[0])
    visualize_data(X_val, Y_val, Y_val)


def print_output_layers(model):
    '''
    Prints the output layers of the given model for both the training and validation datasets.

    Args:
        model: The model object.

    Returns:
        None
    '''
    A, _ = model.nn_forward(model.X_train_norm)
    A_out = A[-1]
    print('=====Train-Output=====')
    print(A_out)
    print('======================\n')
    
    A, _ = model.nn_forward(model.X_val_norm)
    A_out = A[-1]
    print('=====Validation-Output=====')
    print(A_out)
    print('===========================\n')


def train(M, p_val, p_test, sizes, maxepoch, batch_size, K, eta_0, eta_K, file):
    '''
    Trains a neural network model.

    Args:
        M (int): Number of hidden units in the neural network.
        p_val (float): Proportion of validation data.
        p_test (float): Proportion of test data.
        sizes (list): List of integers representing the sizes of the layers in the neural network.
        maxepoch (int): Maximum number of epochs for training.
        batch_size (int): Size of each mini-batch for training.
        K (int): Number of mini-batches.
        eta_0 (float): Initial learning rate.
        eta_K (float): Final learning rate.
        file (str): File path to read the data from.

    Returns:
        model: Trained neural network model.
    '''
    print('Initializing Model...')
    model = nn_model(M, p_val, p_test, sizes, file)

    print('Training model...')
    model.nn_train(batch_size, maxepoch, K, eta_0, eta_K)

    model.visualize_cost_progress()

    # Run model on test data
    A, _ = model.nn_forward(model.X_test_norm)
    A_out = A[-1]
    cost = model.nn_cost(model.Y_test, A_out)
    sccss = model.nn_successrate(model.Y_test, A_out)
    print(f'Result on Test-Data: Success Rate: {sccss:.4f}, Cost: {cost:.4f}')

    # Sample results and visualize them
    original_Y = model.nn_one_hot_inv(model.Y_test)
    inv_hot_out= model.nn_one_hot_inv(A_out)
    sample_X, sample_Y, sample_out = combine_and_sample(model.X_test, original_Y, inv_hot_out)

    visualize_data(sample_X, sample_Y, sample_out)
    return model


def main():
    '''
    This is the main function of the project. It performs various tasks such as setting the file path,
    running tests, training a model, and printing the output layers.

    Parameters:
        None

    Returns:
        None
    '''
    # File must be in project folder or else the path needs to be adapted
    file = '0_9_Handwritten_Data.dat' 
    path = os.path.join(os.path.dirname(__file__), file)
   
    # Run tests only separately:

    test_generate_data(path)
    #test_nn_build_and_one_hot(file)
    #test_gradient(file)

    model = train(10000, 0.1, 0.1, [784, 20, 10], 30, 2, 100, 10e-1, 10e-3, file)
    print_output_layers(model)


if __name__ == "__main__":
    main()
