import tensorflow as tf

def binary_cross_entropy_discriminator_loss(model_output, 
                                        real_label_smoothing_factor=0.9,
                                        fake_label_smoothing_factor=0.1):
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    y_real = model_output['discrimination_on_real_images']
    y_fake = model_output['discrimination_on_generated_images']
    return bce(tf.ones_like(y_real)*real_label_smoothing_factor, y_real) \
         + bce(tf.zeros_like(y_fake)+fake_label_smoothing_factor, y_fake)

def binary_cross_entropy_generator_loss(model_output):
    y_fake = model_output['discrimination_on_generated_images']
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.ones_like(y_fake), y_fake)