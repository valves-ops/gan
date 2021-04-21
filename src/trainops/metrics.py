from tensorflow_gan.examples.mnist import util as eval_util

def frechet_distance(gan_model):
    return eval_util.mnist_frechet_distance(gan_model.real_images, gan_model.generated_images)