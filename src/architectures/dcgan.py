import tensorflow as tf

@tf.function
def dcgan_train_step(batch, model_struct):
    generator = model_struct['generator']['model']
    generator_optimizer = model_struct['generator']['optimizer']
    generator_loss_function = model_struct['generator']['loss']

    discriminator = model_struct['discriminator']['model']
    discriminator_optimizer = model_struct['discriminator']['optimizer']
    discriminator_loss_function = model_struct['discriminator']['loss']

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
            'discrimination_on_generated_images': discrimination_on_generated_images,
            'discriminator_feature_on_generated_images': discriminator_feature_on_generated_images,
            'discrimination_on_real_images': discrimination_on_real_images,
            'discriminator_feature_on_real_images': discriminator_feature_on_real_images,
        }

        generator_loss = generator_loss_function(model_output)
        discriminator_loss = discriminator_loss_function(model_output)

    generator_gradient = tape.gradient(generator_loss, generator.trainable_variables)
    discriminator_gradient = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    del tape

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

def evaluate_capacity_per_layer(capacity_profile, total_capacity, depth):
    raise NotImplementedError

def evaluate_dimensions_per_layer(capacity_per_layer, initial_dimension, final_dimension, progression, progression_morphology):
    raise NotImplementedError

def evaluate_filter_depth_per_layer(capacity_per_layer, dimensions_per_layer):
    raise NotImplementedError

def convolutional_layer_parameters(dimensions_per_layer):
    raise NotImplementedError

def deconvolutional_layer(previous_layer, filter_depth, stride, kernel_size, padding, activation):
    raise NotImplementedError


def build_dcgan_generator(progression, progression_morphology, capacity_profile, 
                          total_capacity, depth, initial_dimension, generated_image_dimension,
                          latent_space_dimension):
    """
        progression: the linear coefficient that determines how rapidly the dimensions of layers are increased
        progression_morphology: the function morphology of the dimension progression
        capacity_profile: how capacity (number of parameters) is distributed across the layers (linear incresing, constant, linear decreasing)
        total_capacity: total parameter count of the convolutional/deconvolutional layers
        depth: number of layers
        initial_dimension: dimension of the first layer
        generated_image_dimension: dimension of the generated image, ie dimension of the final layer
        latent_space_dimension: dimension of the latent space
    """

    #
    capacity_per_layer = evaluate_capacity_per_layer(capacity_profile, total_capacity, depth)

    dimensions_per_layer = evaluate_dimensions_per_layer(capacity_per_layer, initial_dimension, final_dimension, progression, progression_morphology)

    filter_depth_per_layer = evaluate_filter_depth_per_layer(capacity_per_layer, dimensions_per_layer)

    #
    # Kernel Size will be fixed and GIN file defined
    stride_per_layer, padding_per_layer = evaluate = convolutional_layer_parameters(dimensions_per_layer)

    # Build Model
    # Input Layer
    input_layer = keras.layers.Input(latent_space_dimension)
    dense_layer = reshaped_dense_layer(input_layer)
    previous_layer = dense_layer

    for layer in range(depth):
        if layer == depth-1:
            activation = 'tanh' # fixed
        else:
            activation = 'relu' # or another
        deconv_layer = deconvolutional_layer(
                            previous_layer=previous_layer,
                            filter_depth=filter_depth_per_layer[layer],
                            stride=stride_per_layer[layer],
                            # kernel_size, set by GIN file
                            padding=padding_per_layer[layer],
                            activation=activation
                        )
        previous_layer = deconv_layer
    
    output_layer = previous_layer

    generator = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return generator



