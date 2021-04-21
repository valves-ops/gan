class ModelComponent:
    def __init__(self, model, slug):
        self.model = model
        self.slug = slug


class GANModel:
    trainable_components = []
    latent_vectors_batch = None
    real_images = None
    generated_images = None
    discrimination_on_generated_images = None
    discrimination_on_real_images = None
    discriminator_feature_on_generated_images = None
    discriminator_feature_on_real_images = None

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.trainable_components.append(
            ModelComponent(model=self.generator, slug="generator")
        )
        self.discriminator = discriminator
        self.trainable_components.append(
            ModelComponent(model=self.discriminator, slug="discriminator")
        )

    def evaluate(self, batch):
        self.latent_vectors_batch = self._generate_latent_vectors_batch(batch)
        self.generated_images = self._generate_images(self.latent_vectors_batch)
        (
            self.discrimination_on_generated_images,
            self.discriminator_feature_on_generated_images,
        ) = self._discriminate_images(self.generated_images)
        self.real_images = batch[1]
        (
            self.discrimination_on_real_images,
            self.discriminator_feature_on_real_images,
        ) = self._discriminate_images(self.real_images)

    def predict(self, input):
        self.generated_images = self._generate_images(input)
        return self.generated_images

    def _generate_latent_vectors_batch(self, batch):
        return batch[0]

    def _generate_images(self, latent_vectors_batch):
        return self.generator(latent_vectors_batch)

    def _discriminate_images(self, images):
        return self.discriminator(images)
