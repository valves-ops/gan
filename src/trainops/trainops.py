import os
import time

import tensorflow as tf
import tensorflow_gan as tfgan
import numpy as np


class GANTrainOps:
    def __init__(
        self,
        gan_estimator,
        input_function,
        model_slug,
        epochs=100,
        epochs_per_checkpoint=5,
        batches_per_evaluation=500,
        batch_count_for_evalution=10,
    ):
        self.gan_estimator = gan_estimator
        self.input_function = input_function
        self.model_slug = model_slug
        self.epochs = epochs
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.batches_per_evaluation = batches_per_evaluation
        self.batch_count_for_evalution = batch_count_for_evalution

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.gan_estimator.checkpoint,
            self._get_checkpoint_directory(),
            max_to_keep=3,
        )

    def train(self):
        start_time = time.time()

        base_images_latent_vectors = self.get_or_create_base_images_latent_vectors()

        # Create metrics buffers
        metrics_buffer = {}
        for metric_name in self.gan_estimator.evaluation_metrics:
            metrics_buffer.update({metric_name: []})

        for epoch in range(self.epochs):
            batches = self.input_function(mode="train")
            batch_number = 0
            batch_durations = []
            for batch in batches:
                batch_number += 1

                batch_start_time = time.time()

                self.gan_estimator.train_step(batch)

                batch_end_time = time.time()
                batch_durations.append(batch_end_time - batch_start_time)

                if batch_number % self.batches_per_evaluation == 0:
                    # Evaluate
                    # metrics = self.gan_estimator.evaluate(self.input_function, self.batch_count_for_evalution)

                    # Console Logging
                    print(
                        "Time since start: %.2f min"
                        % ((time.time() - start_time) / 60.0)
                    )
                    print(f"Batch Number {batch_number} @ Epoch {epoch}")
                    print(f"Average batch time: {np.average(batch_durations)} secs")

                    # Metrics Logging
                    # for metric_name, metric in metrics.items():
                    #     metrics_buffer[metric_name].append(metric.result())
                    #     print(f'{metric_name}: {metric.result()}')

                    # Generate Images
                    images = self.gan_estimator.predict(base_images_latent_vectors)
                    img_grid = tfgan.eval.python_image_grid(images, grid_shape=(2, 10))
                    plt.axis("off")
                    plt.imshow(np.squeeze(img_grid))
                    plt.show()

                if epoch % self.epochs_per_checkpoint == 0:
                    self.gan_estimator.checkpoint.save(
                        file_prefix=self._get_checkpoint_prefix()
                    )
                    # TODO: Save metrics to disk

    def _get_base_dir(self):
        return os.path.join(os.getcwd(), "data")

    def _get_checkpoint_directory(self):
        return os.path.join(self._get_base_dir(), self.model_slug, "checkpoints")

    def _get_checkpoint_prefix(self):
        return os.path.join(self._get_checkpoint_directory(), "ckpt")

    def restore_latest_checkpoint(self):
        self.gan_estimator.restore(self.checkpoint_manager.latest_checkpoint)

    def _get_base_images_latent_vectors_file_path(self):
        return os.path.join(
            self._get_base_dir(),
            "latent_vectors",
            self.model_slug + "_base_images_latent_vectors.npy",
        )

    def get_or_create_base_images_latent_vectors(self):
        # Try to retrieve from disk
        try:
            base_image_latent_vectors = np.load(
                self._get_base_images_latent_vectors_file_path()
            )

        # Create and save to disk, otherwise
        except FileNotFoundError:
            batch_size = 20
            latent_space_dim = 100
            base_image_latent_vectors = tf.random.normal([batch_size, latent_space_dim])
            filename = self._get_base_images_latent_vectors_file_path()
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(filename, "wb") as f:
                np.save(f, base_image_latent_vectors.numpy())

        return base_image_latent_vectors