import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import tensorflow as tf

from architectures import mnist_dcgans
from model.gan_model import GANModel
from estimator.estimator import GANEstimator
from estimator.losses import binary_cross_entropy_discriminator_loss, binary_cross_entropy_generator_loss
from trainops.metrics import frechet_distance
from trainops.trainops import GANTrainOps


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def input_fn(mode):
    # assert 'batch_size' in params
    # assert 'noise_dims' in params
    bs = 128 #params['batch_size']
    nd = 100 #params['noise_dims']
    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)

    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf.random.normal([bs, nd])))

    if just_noise:
        return noise_ds

    def _preprocess(element):
        # Map [0, 255] to [-1, 1].
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        return images

    images_ds = (tfds.load('mnist:3.*.*', split=split)
                .map(_preprocess)
                .cache()
                .repeat())
    if shuffle:
        images_ds = images_ds.shuffle(
            buffer_size=10000, reshuffle_each_iteration=True)
        images_ds = (images_ds.batch(bs, drop_remainder=True)
                    .prefetch(tf.data.experimental.AUTOTUNE))

    return tf.data.Dataset.zip((noise_ds, images_ds))

def main():
    gan_model = GANModel(
        generator=mnist_dcgans.build_mnist_generator(),
        discriminator=mnist_dcgans.build_mnist_discriminator()
    )

    gan_estimator = GANEstimator(
        gan_model=gan_model,
        component_losses={
            'generator' : binary_cross_entropy_generator_loss,
            'discriminator' : binary_cross_entropy_discriminator_loss,
        },
        component_optimizers={
            'generator' : tf.keras.optimizers.Adam(0.0002, 0.5),
            'discriminator' : tf.keras.optimizers.Adam(0.0002, 0.5),
        },
        evaluation_metrics={
            'frechet_distance' :  frechet_distance
        },
    )

    gan_trainops = GANTrainOps(
        gan_estimator=gan_estimator,
        input_function=input_fn,
        model_slug='mnist-vanilla',
        epochs=20,
    )

    gan_trainops.train()

if __name__ == '__main__':
    main()
