import tensorflow as tf
import gin

import numpy as np
import math


from .generator.dimension_parametrization import (
    evaluate_dimensions_per_layer,
    convert_padding_tf_argument,
)
from .generator.filters_parametrization import evaluate_filter_depth_per_layer


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
