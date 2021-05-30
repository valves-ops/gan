import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import tensorflow as tf

import gin

from architectures import mnist_dcgans, dcgan
from model.gan_model import GANModel
from estimator.estimator import GANEstimator
from trainops.metrics import frechet_distance
from trainops.trainops import GANTrainOps

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _preprocess(element):
        # Map [0, 255] to [-1, 1].
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        return images

@gin.configurable
def get_dataset(BATCH_SIZE = 128, BUFFER_SIZE = 10000, NOISE_DIM = 100):
    print('--- Dataset Setup ---')
    print('BATCH SIZE: ', BATCH_SIZE)
    print('BUFFER SIZE: ', BUFFER_SIZE)
    print('NOISE DIM: ', NOISE_DIM)
    print('---------------------')
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat().map(lambda _: tf.random.normal([BATCH_SIZE, NOISE_DIM])))

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 
    # train_images = (train_images) / 255

    image_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return tf.data.Dataset.zip((noise_ds, image_ds))

def main(gin_filename):
    gin.parse_config_file(gin_filename)
    gan_model = GANModel(
        generator=dcgan.build_dcgan_generator(),
        discriminator=dcgan.build_dcgan_discriminator(),
        train_step_function=dcgan.dcgan_train_step
    )

    gan_estimator = GANEstimator(
        gan_model=gan_model,
        evaluation_metrics={
            'frechet_distance' :  frechet_distance
        },
    )

    gan_trainops = GANTrainOps(
        gan_estimator=gan_estimator,
        dataset=get_dataset(),
    )

    gan_trainops.train()

    return gan_estimator

if __name__ == '__main__':
    gin.external_configurable(tf.keras.optimizers.Adam)
    main('template.gin')
