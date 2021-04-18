import tensorflow as tf


class GANEstimator:
    def __init__(self, 
                 gan_model,
                 component_losses, # { 'component_slug' : loss_fn }
                 component_optimizers, # { 'component_slug' : optimizer }
                 evaluation_metrics, # {'metric_name' : }
                ):
        self.gan_model = gan_model
        self.component_losses = component_losses
        self.component_optimizers = component_optimizers
        self.evaluation_metrics = evaluation_metrics
        self.losses_history = {}

        component_models = []
        for component in self.gan_model.trainable_components:
            self.losses_history[component.slug] = []
            component_models.append(component.model)

        self.checkpoint = tf.train.Checkpoint(
            *(component_models+component_optimizers.values())
        )


    def train_step(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            self.gan_model.evaluate(batch)

            losses = []
            for component in self.gan_model.trainable_components:
                loss = self.component_losses[component.slug](self.gan_model)
                
                self.losses_history[component.slug].append(loss)

        for component, loss in zip(self.losses, self.gan_model.trainable_components):
            gradients = tape.gradient(loss, component.model.trainable_variables)

            self.component_optimizers[component.slug].apply_gradients(
                zip(gradients, component.model.variables))
        
        del tape

    def evaluate(self, input_function, batch_count):
        metrics = {}
        for metric_name in self.evaluation_metrics:
            metrics.update({metric_name: tf.keras.metrics.Mean()})

        for count, batch in enumerate(input_function('train')):
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


