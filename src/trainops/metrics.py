from tensorflow_gan.examples.mnist import util as eval_util
from tensorflow_gan.python.eval.inception_metrics import  frechet_inception_distance

def frechet_distance(gan_model):
    return eval_util.mnist_frechet_distance(gan_model.real_images, gan_model.generated_images)


def general_frechet_distance(gan_model):
    return frechet_inception_distance(gan_model.real_images, gan_model.generated_images)


def get_frechet_distance_func(dataset):
    if dataset == 'MNIST':
        return frechet_distance
    else:
        return general_frechet_distance