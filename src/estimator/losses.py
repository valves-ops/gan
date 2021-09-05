import tensorflow as tf
import gin

@gin.configurable
def binary_cross_entropy_discriminator_loss(model_output, 
                                        real_label_smoothing_factor=0.9):
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    y_real = model_output['discrimination_on_real_images']
    y_fake = model_output['discrimination_on_generated_images']
    return bce(tf.ones_like(y_real)*real_label_smoothing_factor, y_real) \
         + bce(tf.zeros_like(y_fake), y_fake)

@gin.configurable
def binary_cross_entropy_generator_loss(model_output):
    y_fake = model_output['discrimination_on_generated_images']
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.ones_like(y_fake), y_fake)

@gin.configurable
def feature_matching_generator_loss(model_output):
    discriminator_feature_on_real_images = model_output['discriminator_feature_on_real_images']
    discriminator_feature_on_generated_images = model_output['discriminator_feature_on_generated_images']

    avg_discriminator_feature_on_real_images = tf.reduce_mean(discriminator_feature_on_real_images, 0)
    avg_discriminator_feature_on_generated_images = tf.reduce_mean(discriminator_feature_on_generated_images, 0)

    feature_matching_loss = tf.norm(avg_discriminator_feature_on_real_images - avg_discriminator_feature_on_generated_images)

    return feature_matching_loss
