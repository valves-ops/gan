import tensorflow as tf

from anytree import Node, RenderTree, NodeMixin
import anytree
import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar


@tf.function
def dcgan_train_step(batch, model_struct):
    generator = model_struct["generator"]["model"]
    generator_optimizer = model_struct["generator"]["optimizer"]
    generator_loss_function = model_struct["generator"]["loss"]

    discriminator = model_struct["discriminator"]["model"]
    discriminator_optimizer = model_struct["discriminator"]["optimizer"]
    discriminator_loss_function = model_struct["discriminator"]["loss"]

    with tf.GradientTape(persistent=True) as tape:
        latent_vectors_batch = batch[0]
        generated_images = generator(latent_vectors_batch)
        (
            discrimination_on_generated_images,
            discriminator_feature_on_generated_images,
        ) = discriminator(generated_images)
        real_images = batch[1]
        (
            discrimination_on_real_images,
            discriminator_feature_on_real_images,
        ) = discriminator(real_images)

        model_output = {
            "discrimination_on_generated_images": discrimination_on_generated_images,
            "discriminator_feature_on_generated_images": discriminator_feature_on_generated_images,
            "discrimination_on_real_images": discrimination_on_real_images,
            "discriminator_feature_on_real_images": discriminator_feature_on_real_images,
        }

        generator_loss = generator_loss_function(model_output)
        discriminator_loss = discriminator_loss_function(model_output)

    generator_gradient = tape.gradient(generator_loss, generator.trainable_variables)
    discriminator_gradient = tape.gradient(
        discriminator_loss, discriminator.trainable_variables
    )
    del tape

    generator_optimizer.apply_gradients(
        zip(generator_gradient, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradient, discriminator.trainable_variables)
    )


###############
class Convolution(NodeMixin):
    def __init__(self, s, k, p, op, i=None, parent=None, children=None):
        if parent:
            i = parent.o
        o = (i - 1) * s + k - 2 * p + op
        self.name = str(o)
        self.o = int(o)
        self.s = int(s)
        self.k = int(k)
        self.p = int(p)
        self.op = int(op)
        self.parent = parent
        if children:
            self.children = children


def generate_dimension_tree(input_dimension, target_dimension, depth):
    s_space = [1, 2, 3]
    k_space = [5]
    p_space = [(lambda x: 0), (lambda x: math.floor(x / 2))]
    op_space = [0, 1]

    root_node = Convolution(0, 0, 0, 0, i=input_dimension)
    root_node.o = input_dimension
    layers = [[root_node]]
    for layer in range(depth):
        layer_nodes = []
        for parent_node in layers[layer]:
            for s in s_space:
                for op in op_space:
                    if op < s:  # for some reason op has to be lower than s
                        for k in k_space:
                            for p in p_space:
                                conv = Convolution(s, k, p(k), op, i=parent_node.o)
                                if (
                                    conv.o >= input_dimension
                                    and conv.o <= target_dimension
                                ):
                                    if (
                                        layer == depth - 1
                                        and conv.o == target_dimension
                                        or layer < depth - 1
                                    ):
                                        conv.parent = parent_node
                                        layer_nodes.append(conv)
        layers.append(layer_nodes)

    return layers


def integrate_dimension_profile(dimension_profile):
    return integrate.simps(dimension_profile)


def dimension_profile_kurtosis(
    dimension_profile, initial_dimension, target_dimension, depth
):
    a = initial_dimension * depth
    b = target_dimension * depth
    m = (a + b) / 2
    d = (b - a) / 2
    integral = integrate_dimension_profile(dimension_profile)
    kurtosis = (integral - m) / d
    return kurtosis


def get_sorted_dimension_profiles_with_kurtosis(
    layers, initial_dimension, target_dimension, depth
):
    downwards = 2
    root_node = layers[0][0]
    leaf_nodes_w_target_dim = anytree.search.findall_by_attr(
        root_node, str(target_dimension)
    )

    w = anytree.walker.Walker()
    paths_from_root_to_leaf_nodes = [
        w.walk(root_node, leaf) for leaf in leaf_nodes_w_target_dim
    ]

    dimension_profiles = [
        ([initial_dimension] + [layer.o for layer in path[downwards]], path[downwards])
        for path in paths_from_root_to_leaf_nodes
    ]
    dimension_profiles = list(
        filter(lambda x: len(x[0]) == depth + 1, dimension_profiles)
    )

    dimension_profiles_kurtosis = [
        (
            dimension_profile_kurtosis(
                dimension_profile[0], initial_dimension, target_dimension, depth
            ),
            dimension_profile[1],
        )
        for dimension_profile in dimension_profiles
    ]

    sorted_dimension_profiles_kurtosis = sorted(
        dimension_profiles_kurtosis, key=(lambda x: x[0])
    )

    return sorted_dimension_profiles_kurtosis


def get_dimension_profile_with_closest_kurtosis(
    kurtosis, sorted_dimension_profiles_kurtosis
):
    sorted_kurtosis = [
        dimension_profile[0] for dimension_profile in sorted_dimension_profiles_kurtosis
    ]
    sorted_kurtosis = np.array(sorted_kurtosis)
    idx = sorted_kurtosis.searchsorted(kurtosis)
    idx = np.clip(idx, 1, len(sorted_kurtosis) - 1)
    left = sorted_kurtosis[idx - 1]
    right = sorted_kurtosis[idx]
    idx -= kurtosis - left < right - kurtosis
    return sorted_dimension_profiles_kurtosis[idx]


def convert_padding_tf_argument(padding):
    if padding == 0:
        return "valid"
    else:
        return "same"


def evaluate_dimensions_per_layer(kurtosis, initial_dimension, target_dimension, depth):
    layers = generate_dimension_tree(initial_dimension, target_dimension, depth)
    sorted_dimension_profiles_kurtosis = get_sorted_dimension_profiles_with_kurtosis(
        layers, initial_dimension, target_dimension, depth
    )
    dimension_profile = get_dimension_profile_with_closest_kurtosis(
        kurtosis, sorted_dimension_profiles_kurtosis
    )
    layer_nodes = dimension_profile[1]
    return layer_nodes


#############
def layer_capacity(kernel_dim, channels, filters, bias=True):
    conv_layer_capacity = kernel_dim * kernel_dim * filters * channels + bias * filters
    batch_norm_capacity = 4 * filters
    layer_capacity = conv_layer_capacity + batch_norm_capacity
    return layer_capacity


def calculate_network_capacity(
    kernel_dim, filters_profile, dimensions_per_layer, latent_space_dimension
):
    first_conv_layer_dimensions_input = dimensions_per_layer[0].parent.o
    dense_layer_output_size = (
        first_conv_layer_dimensions_input
        * first_conv_layer_dimensions_input
        * filters_profile[0]
    )
    dense_layer_capacity = dense_layer_output_size * (latent_space_dimension + 1)

    dense_layer_batch_norm_capacity = 4 * dense_layer_output_size
    TC = dense_layer_capacity + dense_layer_batch_norm_capacity
    previous_layers_channel = filters_profile[0]
    for filter in filters_profile:  # [1:]:
        TC += layer_capacity(kernel_dim, previous_layers_channel, filter, bias=True)
        previous_layers_channel = filter
    return TC


def get_filters_profile(kurtosis, depth, initial_size, target_size):
    x1 = depth - 1
    y1 = target_size
    y2 = initial_size
    delta = (y1 - y2) / x1
    alpha = kurtosis * delta / x1
    beta = (y1 - y2) / x1 - alpha * x1
    gama = y2
    x = np.array(range(depth))
    y = alpha * x ** 2 + beta * x + gama
    y = np.ceil(y).astype(int)
    return y


def capacity_opt_function(
    initial_filter_depth,
    target_capacity,
    kernel_dim,
    depth,
    kurtosis,
    target_filter_depth,
    dimensions_per_layer,
    latent_space_dimension,
):
    filters_profile = get_filters_profile(
        kurtosis,
        depth,
        initial_filter_depth,
        target_filter_depth,
    )
    calculated_capacity = calculate_network_capacity(
        kernel_dim, filters_profile, dimensions_per_layer, latent_space_dimension
    )
    return np.abs(target_capacity - calculated_capacity)


def evaluate_filter_depth_per_layer(
    total_capacity,
    kernel_dim,
    depth,
    kurtosis,
    target_filter_depth,
    dimensions_per_layer,
    latent_space_dimension,
):
    result = minimize_scalar(
        capacity_opt_function,
        bounds=(1, 2048),
        args=(
            total_capacity,
            kernel_dim,
            depth,
            kurtosis,
            target_filter_depth,
            dimensions_per_layer,
            latent_space_dimension,
        ),
        method="bounded",
    )
    estimated_initial_filter_depth = result.x

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
        kurtosis, depth, estimated_initial_filter_depth, target_filter_depth
    )
    return filters_profile


############
def reshaped_dense_layer(previous_layer, dimension, filters):
    dense_layer = tf.keras.layers.Dense(
        units=dimension * dimension * filters, activation=tf.nn.relu
    )(previous_layer)
    batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(dense_layer)
    activation = tf.keras.layers.Activation("relu")(batch_norm_layer)
    reshape_layer = tf.keras.layers.Reshape((dimension, dimension, filters))(activation)
    return reshape_layer


def convolutional_layer_parameters(dimensions_per_layer):
    raise NotImplementedError


def deconvolutional_layer(
    previous_layer,
    filter_depth,
    stride,
    kernel_size,
    padding,
    output_padding,
    activation,
):
    deconv_layer = tf.keras.layers.Conv2DTranspose(
        kernel_size=kernel_size,
        filters=filter_depth,
        strides=stride,
        output_padding=output_padding,
        padding=padding,
    )(previous_layer)
    batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(deconv_layer)
    activation = tf.keras.layers.LeakyReLU(alpha=0.1)(batch_norm_layer)
    return activation


def build_dcgan_generator(
    dimension_progression_kurtosis,
    filters_depth_progression_kurtosis,
    total_capacity,
    depth,
    kernel_dimension,
    initial_dimension,
    target_dimension,
    latent_space_dimension,
):
    """
    dimension_progression_kurtosis: value between -1 and 1 that determines the morphology of dimensions progression ("exponential", "linear", "logarithmic")
    filters_progression_kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    capacity_profile: how capacity (number of parameters) is distributed across the layers (linear incresing, constant, linear decreasing)
    total_capacity: total parameter count of the convolutional/deconvolutional layers
    depth: number of layers
    initial_dimension: dimension of the first layer
    target_dimension: dimension of the generated image, ie dimension of the final layer
    latent_space_dimension: dimension of the latent space
    """
    dimensions_per_layer = evaluate_dimensions_per_layer(
        dimension_progression_kurtosis, initial_dimension, target_dimension[0], depth
    )

    filter_depth_per_layer = evaluate_filter_depth_per_layer(
        total_capacity,
        kernel_dimension,
        depth,
        filters_depth_progression_kurtosis,
        target_dimension[1],
        dimensions_per_layer,
        latent_space_dimension,
    )

    # Build Model
    # Input Layer
    input_layer = tf.keras.layers.Input(latent_space_dimension)
    dense_layer = reshaped_dense_layer(
        input_layer, dimensions_per_layer[0].parent.o, filter_depth_per_layer[0]
    )
    previous_layer = dense_layer

    for layer in range(depth):
        if layer == depth - 1:
            activation = "tanh"  # fixed
        else:
            activation = "relu"  # or another
        deconv_layer = deconvolutional_layer(
            previous_layer=previous_layer,
            filter_depth=filter_depth_per_layer[layer],
            stride=dimensions_per_layer[layer].s,
            kernel_size=kernel_dimension,
            padding=convert_padding_tf_argument(dimensions_per_layer[layer].p),
            output_padding=dimensions_per_layer[layer].op,
            activation=activation,
        )
        previous_layer = deconv_layer

    output_layer = previous_layer

    generator = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return generator
