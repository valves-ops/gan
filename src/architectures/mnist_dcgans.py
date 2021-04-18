from tensorflow import keras
import tensorflow as tf

## Model Parameters
img_input_shape = (28, 28, 1)

latent_space_shape = (100,)

depth = 64+64+64+64
dim = 7
dropout_rate = 0.4

def reshaped_dense_layer(previous_layer):
  dense_layer = keras.layers.Dense(
        units=dim*dim*depth,
        activation=tf.nn.relu)(previous_layer)
  batch_norm_layer = keras.layers.BatchNormalization(momentum=0.9)(dense_layer)
  activation = keras.layers.Activation('relu')(batch_norm_layer)
  reshape_layer = keras.layers.Reshape((dim, dim, depth))(activation)
  return reshape_layer

def convolutional_layer(previous_layer, filter_depth, kernel_size, stride):
  conv_layer = keras.layers.Conv2D(
      filters=filter_depth,
      kernel_size=kernel_size,
      strides=stride,
      padding='same')(previous_layer)
  leaky_relu_layer = keras.layers.LeakyReLU(alpha=0.2)(conv_layer)
  batch_norm_layer = keras.layers.BatchNormalization(momentum=0.9)(leaky_relu_layer)
  dropout_layer = keras.layers.Dropout(rate=dropout_rate)(leaky_relu_layer)
  return batch_norm_layer

def deconvolutional_layer(previous_layer, filter_depth, strides):
  deconv_layer = keras.layers.Conv2DTranspose(
      filters=filter_depth,
      kernel_size=(5, 5),
      strides=strides,
      use_bias=False,
      padding='same')(previous_layer)
  batch_norm_layer = keras.layers.BatchNormalization(momentum=0.9)(deconv_layer)
  activation = keras.layers.LeakyReLU(alpha=0.1)(batch_norm_layer)
  return activation

def build_mnist_generator():
  input_layer = keras.layers.Input(latent_space_shape)
  dense_layer = reshaped_dense_layer(input_layer)
  deconv0 = deconvolutional_layer(dense_layer, 128, 1)
  deconv1 = deconvolutional_layer(deconv0, 64, 2)   
  deconv2 = deconvolutional_layer(deconv1, 32, 1)  
  output_layer = keras.layers.Conv2DTranspose(
      filters=1,
      kernel_size=(5, 5),
      strides=1,
      padding='same',
      activation='tanh')(deconv2)

  generator = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  return generator

def build_mnist_discriminator():
  input_layer = keras.layers.Input(img_input_shape)
  conv1 = convolutional_layer(input_layer, 64, 5, 1)
  conv2 = convolutional_layer(conv1, 128, 5, 2)
  conv3 = convolutional_layer(conv2, 256, 5, 2)
  conv4 = convolutional_layer(conv3, 512, 5, 2)
  flatten_layer = keras.layers.Flatten()(conv4)
  output_layer = keras.layers.Dense(1, activation=keras.activations.sigmoid)(flatten_layer)
  D = tf.keras.Model(inputs=input_layer, outputs=[output_layer, flatten_layer])
  return D