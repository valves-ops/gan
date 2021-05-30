import tensorflow as tf
import gin

import numpy as np
import math

from .generator import filters_parametrization as generator_filters
from .generator import dimension_parametrization as generator_dimensions

from .discriminator import filters_parametrization as discriminator_filters
from .discriminator import dimension_parametrization as discriminator_dimensions


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


def reshaped_dense_layer(previous_layer, dimension, filters):
    dense_layer = tf.keras.layers.Dense(
        units=dimension * dimension * filters, activation=tf.nn.relu
    )(previous_layer)
    batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(dense_layer)
    activation = tf.keras.layers.Activation("relu")(batch_norm_layer)
    reshape_layer = tf.keras.layers.Reshape((dimension, dimension, filters))(activation)
    return reshape_layer


def convolutional_layer(previous_layer,
                        filter_depth,
                        stride,
                        kernel_size,
                        padding,
                        activation,):
    conv_layer = tf.keras.layers.Conv2D(
        kernel_size=kernel_size,
        filters=filter_depth,
        strides=stride,
        padding=padding,
    )(previous_layer)
    batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=0.9)(conv_layer)
    activation = tf.keras.layers.LeakyReLU(alpha=0.1)(batch_norm_layer)
    return activation


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

def convert_padding_tf_argument(padding):
    if padding == 0:
        return "valid"
    else:
        return "same"

@gin.configurable
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
    filters_depth_progression_kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    total_capacity: total parameter count of the convolutional/deconvolutional layers
    depth: number of convolutional layers
    kernel: dimension of the kernel to be used across all layers
    initial_dimension: dimension of the first layer
    target_dimension: dimension of the generated image, ie dimension of the final layer
    latent_space_dimension: dimension of the latent space
    """
    dimensions_per_layer = generator_dimensions.evaluate_dimensions_per_layer(
        dimension_progression_kurtosis, initial_dimension, target_dimension[0], depth
    )

    filter_depth_per_layer = generator_filters.evaluate_filter_depth_per_layer(
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


@gin.configurable
def build_dcgan_discriminator(
    dimension_progression_kurtosis,
    filters_depth_progression_kurtosis,
    total_capacity,
    depth,
    kernel_dimension,
    initial_dimension,
    target_dimension,
):
    """
    dimension_progression_kurtosis: value between -1 and 1 that determines the morphology of dimensions progression ("exponential", "linear", "logarithmic")
    filters_depth_progression_kurtosis: value between -1 and 1 that determines the morphology of filters progression ("exponential", "linear", "logarithmic")
    total_capacity: total parameter count of the network
    depth: number of convolutional layers
    kernel: dimension of the kernel to be used across all layers
    initial_dimension: dimension of the generated image, ie dimension of the input layer, tuple (HxW, depth)
    target_dimension: dimension of the tensor prior to the dense layer
    """

    dimensions_per_layer = discriminator_dimensions.evaluate_dimensions_per_layer(
        dimension_progression_kurtosis, initial_dimension[0], target_dimension, depth
    )

    filter_depth_per_layer = discriminator_filters.evaluate_filter_depth_per_layer(
        total_capacity,
        kernel_dimension,
        depth,
        filters_depth_progression_kurtosis,
        initial_dimension[1],
        dimensions_per_layer
    )

    # Build Model
    # Input Layer
    input_layer = tf.keras.layers.Input((initial_dimension[0], 
                                         initial_dimension[0], 
                                         initial_dimension[1]))
    previous_layer = input_layer

    for layer in range(depth):
        activation = "relu"  # or another
        conv_layer = convolutional_layer(
            previous_layer=previous_layer,
            filter_depth=filter_depth_per_layer[layer],
            stride=dimensions_per_layer[layer].s,
            kernel_size=kernel_dimension,
            padding=convert_padding_tf_argument(dimensions_per_layer[layer].p),
            activation=activation,
        )
        previous_layer = conv_layer

    flatten_layer = tf.keras.layers.Flatten()(previous_layer)
    output_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(flatten_layer)

    discriminator = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return discriminator
