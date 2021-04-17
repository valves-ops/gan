import os
import time

import tensorflow as tf
import numpy as np


class GANTrainOps:
    def __init__(
        self,
        gan_estimator,
        input_function,
        model_slug,
        epochs=100,
        batches_per_checkpoint=100,
        batches_per_log=100,
        batches_per_evaluation=100,
        batch_count_for_evalution=10,
    ):
        self.gan_estimator = gan_estimator
        self.input_function = input_function
        self.model_slug = model_slug
        self.epochs = epochs
        self.batches_per_checkpoint = batches_per_checkpoint
        self.batches_per_evaluation = batches_per_evaluation
        self.batches_per_log = batches_per_log
        self.batch_count_for_evalution = batch_count_for_evalution

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.gan_estimator.checkpoint, 
            self._get_checkpoint_directory, 
            max_to_keep=3
        )

    def train(self):
        start_time = time.time()
        for epoch in self.epochs:
            batches = self.input_function(mode="TRAIN")
            batch_number = 0
            batch_durations = []
            for batch in batches:
                batch_start_time = time.time()

                self.gan_estimator.train_step(batch)

                batch_end_time = time.time()

                batch_number += 1
                batch_durations.append(batch_end_time - batch_start_time)

                if batch_number % self.batches_per_checkpoint:
                    self.gan_estimator.checkpoint.save(
                        file_prefix=self.get_checkpoint_prefix()
                    )

                if batch_number % self.batches_per_evaluation:
                    self.gan_estimator.evaluate(self.input_function, self.batch_count_for_evalution)
                    print(
                        "Time since start: %.2f min"
                        % ((time.time() - start_time) / 60.0)
                    )
                    print(f"Batch Number {batch_number} @ Epoch {epoch}")
                    print(f"Average batch time: {np.average(batch_durations)} secs")

                if batch_number % self.batches_per_log:
                    pass

    def _get_checkpoint_prefix(self):
        return os.path.join(self._get_checkpoint_directory(), "ckpt")

    def _get_checkpoint_directory(self):
        return os.path.join(os.getcwd(), "data", self.model_slug, "checkpoints")

    def restore_latest_checkpoint(self):
        self.gan_estimator.restore(self.checkpoint_manager.latest_checkpoint)
