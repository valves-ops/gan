import tensorflow as tf


class GANEstimator:
    def __init__(
        self,
        gan_model,
        component_losses,  # { 'component_slug' : loss_fn }
        component_optimizers,  # { 'component_slug' : optimizer }
        evaluation_metrics,  # {'metric_name' : }
    ):
        self.gan_model = gan_model
        self.component_losses = component_losses
        self.component_optimizers = component_optimizers
        self.evaluation_metrics = evaluation_metrics
        self.losses_history = {}

        models_and_optimizers = {}
        for component in self.gan_model.trainable_components:
            self.losses_history[component.slug] = []
            models_and_optimizers.update({(component.slug + "_model"): component.model})

        for optimizer_slug, optimizer in self.component_optimizers.items():
            models_and_optimizers.update({(optimizer_slug + "_optimizer"): optimizer})

        self.checkpoint = tf.train.Checkpoint(**models_and_optimizers)

    def train_step(self, batch):
        tf_train_step_function = self.gan_model.train_step_function
        model_struct = {
            key: {
                'model': list(filter(
                    lambda component, slug=key: component.slug == slug,
                    self.gan_model.trainable_components))[0].model,
                'optimizer': self.component_optimizers[key],
                'loss': self.component_losses[key]
            }
            for key in self.component_optimizers.keys()
        }
        tf_train_step_function(batch, model_struct)

    # TODO: Update evaluation function
    def evaluate(self, dataset, batch_count):
        metrics = {}
        for metric_name in self.evaluation_metrics:
            metrics.update({metric_name: tf.keras.metrics.Mean()})

        for count, batch in enumerate(dataset):
            self.gan_model.evaluate(batch)
            # Calculate Metrics
            for metric_name, metric_function in self.evaluation_metrics.items():
                metrics[metric_name].update_state(metric_function(self.gan_model))

        # Return metrics
        return metrics

    def predict(self, input):
        return self.gan_model.predict(input)

    def restore(self, checkpoint):
        self.checkpoint.restore(checkpoint)
