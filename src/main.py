import argparse
import glob
import os

import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import tensorflow as tf

import gin
import wandb

from architectures import mnist_dcgans, dcgan
from model.gan_model import GANModel
from estimator.estimator import GANEstimator
from trainops.metrics import frechet_distance
from trainops.trainops import GANTrainOps
from estimator.losses import binary_cross_entropy_discriminator_loss, binary_cross_entropy_generator_loss, feature_matching_generator_loss


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _preprocess(element):
        # Map [0, 255] to [-1, 1].
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
        return images

@gin.configurable
def get_dataset(BATCH_SIZE = 128, BUFFER_SIZE = 10000, NOISE_DIM = 100):
    wandb.config.batch_size = BATCH_SIZE
    wandb.config.buffer_size = BUFFER_SIZE
    wandb.config.noise_dim = NOISE_DIM
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

@gin.configurable
def main(gin_filename=None, wandb_project='mnist-test-setup'):
    if gin_filename:
        gin.parse_config_file(gin_filename)

    wandb.init(project=wandb_project, name=gin.query_parameter('GANTrainOps.model_slug'))

    gan_model = GANModel(
        generator=dcgan.build_dcgan_generator(),
        discriminator=dcgan.build_dcgan_discriminator(),
        # generator=mnist_dcgans.build_mnist_generator(),
        # discriminator=mnist_dcgans.build_mnist_discriminator(),
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

def get_intermediary_directories(initial_dir, final_dir):
    subdir_diff = final_dir.replace(initial_dir, '')
    subdir_list = subdir_diff.split(os.sep)
    print(subdir_list[:-1])
    return subdir_list

def get_gin_files_list_for_experiment(experiment_path, run_file):
    main_path = os.path.dirname(os.path.realpath(__file__))
    experiments_folder_abs_path = os.path.join(main_path, 'experiments')
    experiment_abs_path = os.path.join(experiments_folder_abs_path, 
                                       *experiment_path.split('.'))
    dirs_to_collect_configs = get_intermediary_directories(
                                    experiments_folder_abs_path, 
                                    experiment_abs_path)
    gin_base_config_files = []
    previous_dir = experiments_folder_abs_path
    for subdir in dirs_to_collect_configs:
        config_file_dir = os.path.join(previous_dir, subdir)
        config_files_list = glob.glob(os.path.join(config_file_dir, '*.gin'))
        if len(config_files_list) > 0:
            gin_base_config_files.append(config_files_list[0])
        previous_dir = config_file_dir

    run_files = glob.glob(os.path.join(experiment_abs_path, 'runs', '*.gin'))
    gin_run_file = os.path.join(experiment_abs_path, 'runs', run_file)
    
    return gin_base_config_files, gin_run_file
    

if __name__ == '__main__':
    gin.external_configurable(tf.keras.optimizers.Adam)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment', type=str)
    argparser.add_argument('--run', type=str)

    args = argparser.parse_args()
    experiment_path = args.experiment
    # gin_run_file = os.path.join('experiments', *experiment_path.split('.'), args.run)
    gin_base_files, gin_run_file = get_gin_files_list_for_experiment(experiment_path, args.run)
    
    print('BASE GIN FILES: ', gin_base_files)
    print('RUN GIN FILE: ', gin_run_file)
    
    iteration_gin_files = gin_base_files + [gin_run_file]
    gin.parse_config_files_and_bindings(config_files=iteration_gin_files, bindings=[])
    main()
        
