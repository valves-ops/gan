
import numpy as np
import math

from scipy.optimize import minimize_scalar


def layer_capacity(kernel_dim, channels, filters, bias=True):
    """
    Calculates the capacity (parameter count) of a deconvolutional layer

    kernel_dim: integer, kernel dimension of the deconvolutional layer
    channels: integer, number of channels of the input tensor
    filters: integer, number of filters of the deconvolutional layer
    bias: boolean, wether the deconvolutional layer uses or not a bias term
    """
    conv_layer_capacity = kernel_dim * kernel_dim * filters * channels + bias * filters
    batch_norm_capacity = 4 * filters
    layer_capacity = conv_layer_capacity + batch_norm_capacity
    return layer_capacity


def calculate_network_capacity(
    kernel_dim, filters_profile, dimensions_per_layer
):
    """
    Calculates the capacity (parameter count) of a entire network

    kernel_dim: integer, kernel dimension used across the network
    filters_profile: list of integers representing the number of filters of each deconvolutional layer
    dimensions_per_layer: list of Convolution Objects representing each deconvolutional layer
    latent_space_dimension: integer represeting the latent space vector length
    """
    last_conv_layer_dimensions_output = dimensions_per_layer[-1].o
    dense_layer_output_size = (
        last_conv_layer_dimensions_output
        * last_conv_layer_dimensions_output
        * filters_profile[-1]
    )
    dense_layer_capacity = dense_layer_output_size + 1

    TC = dense_layer_capacity
    previous_layers_channel = filters_profile[0]
    for filter in filters_profile:  # [1:]:
        TC += layer_capacity(kernel_dim, previous_layers_channel, filter, bias=True)
        previous_layers_channel = filter
    return TC


def get_filters_profile(kurtosis, depth, initial_size, target_size):
    """
    Calculates a list of filter numbers according to a given kurtosis value

    kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    depth: integer representing the number of deconvolutional layers in the generator
    initial_size: amount of filters in the first layer of the generator
    target_size: amount of filters in the last layer of the generator, ie output tensor number of channels
    """
    x1 = depth - 1
    y1 = target_size
    y2 = initial_size
    delta = (y2 - y1) / x1
    alpha = kurtosis * delta / x1
    beta = (y1 - y2) / x1 - alpha * x1
    gama = y2
    x = np.array(range(depth))
    y = alpha * x ** 2 + beta * x + gama
    y = np.ceil(y).astype(int)
    return y


def capacity_opt_function(
    target_filter_depth,
    target_capacity,
    kernel_dim,
    depth,
    kurtosis,
    initial_filter_depth,
    dimensions_per_layer,
):
    """
    Calculates distance between the target capacity of the generator and the capacity that the generator
    would have if constructed with the provided parameters.
    This function is used to optimize the initial_filter_depth parameter

    initial_filter_depth: integer represeting the number of filters in the first deconvolutional layer
    target_capacity: total capacity (parameter count) the generator must have
    kernel_dim: integer, kernel dimension of the deconvolutional layer
    depth: integer representing the number of deconvolutional layers in the generator
    kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    target_filter_depth: integer represeting the number of filters in the last deconvolutional layer
    dimensions_per_layer: list of Convolution Objects representing each deconvolutional layer
    latent_space_dimension: integer represeting the latent space vector length
    """
    filters_profile = get_filters_profile(
        kurtosis,
        depth,
        initial_filter_depth,
        target_filter_depth,
    )
    calculated_capacity = calculate_network_capacity(
        kernel_dim, filters_profile, dimensions_per_layer
    )
    return np.abs(target_capacity - calculated_capacity)


def evaluate_filter_depth_per_layer(
    total_capacity,
    kernel_dim,
    depth,
    kurtosis,
    initial_filter_depth,
    dimensions_per_layer,
):
    """
    Given the total capacity desired for the generator, a kurtosis for the filters progression, the depth of the generator and
    the output tensor number of channels, calculates the filters progression.

    total_capacity: total capacity (parameter count) the generator must have
    kernel_dim: integer, kernel dimension of the deconvolutional layer
    depth: integer representing the number of deconvolutional layers in the generator
    kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    target_filter_depth: integer represeting the number of filters in the last deconvolutional layer, ie the output tensor number of channels
    dimensions_per_layer: list of Convolution Objects representing each deconvolutional layer
    latent_space_dimension: integer represeting the latent space vector length
    """
    result = minimize_scalar(
        capacity_opt_function,
        bounds=(1, 2048),
        args=(
            total_capacity,
            kernel_dim,
            depth,
            kurtosis,
            initial_filter_depth,
            dimensions_per_layer,
        ),
        method="bounded",
    )
    estimated_target_filter_depth = result.x

    # filters_profile = get_filters_profile(
    #     kurtosis,
    #     depth,
    #     estimated_initial_filter_depth,
    #     target_filter_depth,
    # )
    # calculated_capacity = calculate_network_capacity(
    #     kernel_dim, filters_profile, dimensions_per_layer, latent_space_dimension
    # )

    filters_profile = get_filters_profile(
        kurtosis, depth, initial_filter_depth, estimated_target_filter_depth
    )
    return filters_profile