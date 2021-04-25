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
