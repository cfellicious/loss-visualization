import caffe
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_param_count(net=None):
    """
    This function calculates the number of parameters in the Caffe Network
    :param net: The caffe network that has been loaded
    :return: The number of parameters and its Frobenius Norm
    """
    if net is None:
        return None

    param_count = 0
    norm = 0
    layer_names = net.blobs
    for layer in layer_names:
        curr_layer = net.layer_dict.get(layer, None)
        if curr_layer is None:
            continue
        if net.layer_dict.get(layer, None).type in ('Convolution', 'InnerProduct'):
            # TODO: Calculate the weight matrix too
            # TODO: Add bias values too. Currently, only weights are used.
            curr_layer_shape = net.params[layer][0].data.shape
            norm = norm + (net.params[layer][0].data * net.params[layer][0].data).sum()
            layer_params = 1
            for layer_shape in curr_layer_shape:
                layer_params = layer_params * layer_shape

            param_count = param_count + layer_params

    norm = np.sqrt(norm)
    return param_count, norm


def get_gaussian_vector(param_count=0,
                        vector_count=0):
    """
    This function generates a number of random gaussian vectors of length param count
    :param param_count: The number of random values to be generated
    :param vector_count: The number of random gaussian vectors to be generated
    :return: A numpy matrix of random Gaussian vectors
    """
    if param_count == 0 or vector_count == 0:
        return 0
    # This samples from a uniform distribution between 0 and 1, with param_count columns
    # and vector_count rows
    return np.random.uniform(low=0, high=1, size=(vector_count, param_count))


def calculate_norm(input_matrix=None):
    """
    This function calculates the Frobenius norm of an input matrix
    :param input_matrix: The input matrix of which the Frobenius norm is to be calculated
    This function assumes that each row is an independent vector and calculates the norm for
    each row
    :return: The Frobenius norm of each row in the matrix
    """
    if input_matrix is None:
        return 0

    return np.sqrt(np.sum(input_matrix * input_matrix, axis=1))


def create_grid(vectors=None, steps=0):
    """
    This function creates a square grid of values between the starting and ending vectors defined by steps
    :param vectors: The starting and ending vectors
    :param steps: The number of steps from the starting to ending values
    :return: A matrix of size steps X number of columns in the vector
    """
    if vectors is None or steps == 0:
        return None

    # Get the shape of the vectors
    vector_shape = list(np.shape(vectors))
    if vector_shape[0] != 2:
        return None

    # Create the range of values for each value in the vector with the initial value
    # at row 0 and final value at row 1 for any column in the vectors
    value_matrix = np.linspace(vectors[0][0], vectors[1][0], num=steps).reshape(steps, 1)
    for col in range(1, vector_shape[1]):
        value_matrix = np.hstack([value_matrix, \
                                  np.linspace(vectors[0][col], vectors[1][col], num=steps).reshape(steps, 1)])

    return value_matrix


def update_net_params(net, vector1, vector2):
    layer_names = net.blobs
    for layer in layer_names:
        curr_layer = net.layer_dict.get(layer, None)
        if curr_layer is None:
            continue

        vec_idx = 0
        if net.layer_dict.get(layer, None).type in ('Convolution', 'InnerProduct'):
            layer_length = np.shape(net.params[layer][0].data.flat)[0]
            net.params[layer][0].data.flat = \
                net.params[layer][0].data.flat + \
                vector1[vec_idx:vec_idx + layer_length] + \
                vector2[vec_idx:vec_idx+layer_length]
            vec_idx = vec_idx + layer_length


def create_loss_landscape(net=None, vectors=None):
    """
    This function creates a grid with the vectors to visualize the loss landscape of the network
    :param net: The Neural Network whose loss landscape is to be visualized
    :param vectors: The normalized gaussian vectors
    :return: A matrix containing the loss values for each point in the grid
    """
    steps = 15
    start_time = time.time()
    vector_grid1 = create_grid(np.vstack((vectors[0], np.negative(vectors[0]))), steps)
    vector_grid2 = create_grid(np.vstack((vectors[1], np.negative(vectors[1]))), steps)
    end_time = time.time() - start_time
    print('Duration : ' + str(end_time))

    # Load a default image and set it as the data
    im = caffe.io.load_image('/home/chris/PycharmProjects/loss-visualization/airplane1.png')
    net.blobs['data'] = np.asarray(im)

    print('Default Loss:', net.forward())

    loss_matrix = np.zeros((steps, steps))
    for x_idx in range(0, steps):
        for y_idx in range(0, steps):
            print(x_idx, y_idx)
            # Modify the network values
            update_net_params(net, vector_grid1[x_idx, :], vector_grid2[y_idx, :])
            # Calculate the loss
            loss = net.forward()
            # Save the loss value to a matrix
            loss_matrix[x_idx][y_idx] = loss.get('loss', 0)

    return loss_matrix


def main():
    # Load the network into memory through the pyCaffe Interface
    # TODO: The solver and prototxt files should be chosen through GUI
    net = caffe.Net('/home/chris/caffe/python/solver.prototxt',
                    '/home/chris/caffe/python/model.caffemodel', caffe.TRAIN)

    # Calculate the total parameter count in the network
    # Calculate the Frobenius norm/Euclidean norm of the network
    # Square root of sum of absolute squares of all the weights in the network
    param_count, euclidean_norm = calculate_param_count(net)

    # Get the normalized Gaussian vectors for the total number of parameters
    # Vector count is currently 2 because only two vectors are needed in x and y directions
    vector_count = 2
    gaussian_vec = get_gaussian_vector(param_count, vector_count)

    # Normalize the Gaussian Vector with the norm
    vectors_norms = calculate_norm(gaussian_vec)
    normalized_vectors = np.divide(gaussian_vec, np.reshape(vectors_norms, [len(vectors_norms), 1]))

    # Multiply the vectors with the norm of the Network
    directional_vectors = np.multiply(normalized_vectors, euclidean_norm)

    loss_values = create_loss_landscape(net, directional_vectors)

    print(np.asarray(loss_values))

    x = y = np.arange(-3.5, 4.0, 0.5)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, loss_values)
    plt.show()


if __name__ == "__main__":
    main()
