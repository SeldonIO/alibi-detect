import tensorflow as tf
from keras import backend as K
import os
from functools import reduce
import numpy as np
import keras


def sampling_gaussian(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def sampling_nlp(args):
    """Sampling from discrete distribution.
    Arguments
        args: probabilities of each category, input dimension
    Return
        sequence of lenght input_dim of categorical values sampled with probabilities p
    """
    p, input_dim = args
    return tf.random.categorical(tf.log(p), input_dim)


class VaeSymmetryFinder(object):
    """Variational Autoencoder designed to find model's symmetries
    """
    def __init__(self, predict_fn, input_shape=(28, 28), output_shape=(10, ),
                 intermediate_dim=5, latent_dim=2, input_dtype="float32",
                 intermediate_activation='relu', output_activation='relu', opt='Adam', lr=0.001):
        self.predict_fn = predict_fn
        self.input_shape = input_shape
        if len(self.input_shape) > 1:
            self.flatten_dim = reduce(lambda v, w: v * w, input_shape)
        else:
            self.flatten_dim = self.input_shape[0]
        self.original_dim = (self.flatten_dim,)
        self.output_shape = output_shape
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.output_activation = output_activation
        self.latent_dim = latent_dim
        self.opt = opt
        self.lr = lr

        # It works for keras models only for now
        if isinstance(self.predict_fn, tf.keras.models.Model) or isinstance(self.predict_fn, keras.models.Model):
            for layer in self.predict_fn.layers:
                layer.trainable = False
        else:
            raise NotImplementedError

        self.inputs = tf.keras.layers.Input(shape=self.input_shape, dtype=input_dtype, name='encoder_input')
        self.x = tf.keras.layers.Reshape(target_shape=self.original_dim)(self.inputs)

        self.x = tf.keras.layers.Dense(self.intermediate_dim, activation=self.intermediate_activation)(self.x)

        self.z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(self.x)
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(self.x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = tf.keras.layers.Lambda(sampling_gaussian, output_shape=(self.latent_dim,),
        name='z')([self.z_mean, self.z_log_var])

        self.x = tf.keras.layers.Dense(self.intermediate_dim, activation=self.intermediate_activation)(self.z)
        self.vae_outputs = tf.keras.layers.Dense(self.flatten_dim, activation=self.output_activation)(self.x)
        self.vae_outputs = tf.keras.layers.Reshape(target_shape=self.input_shape, dtype=input_dtype,
                                                   name='input_trans')(self.vae_outputs)

        self.model_output_trans = self.predict_fn(self.vae_outputs)
        self.model_output_orig = self.predict_fn(self.inputs)

        self.vae = tf.keras.models.Model(self.inputs, [self.vae_outputs, self.model_output_orig,
                                                       self.model_output_trans], name='vae_mlp')

        self.loss = tf.keras.losses.kullback_leibler_divergence(self.model_output_orig, self.model_output_trans)
        self.vae_loss = K.mean(self.loss)
        self.vae.add_loss(self.vae_loss)
        if self.opt == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        elif self.opt == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(self.lr)
        self.vae.compile(optimizer=self.optimizer)
        print('Vae')
        self.vae.summary()

    def fit(self, X_train, x_test=None, epochs=2, batch_size=128):
        if x_test is not None:
            self.vae.fit(X_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_test, None))
        else:
            self.vae.fit(X_train,
                         epochs=epochs,
                         batch_size=batch_size)

    def save(self, save_dir=''):
        file_arch = os.path.join(save_dir, 'vae_arch.json')
        file_weights = os.path.join(save_dir, 'vae_weights.h5')
        json_model = self.vae.to_json()
        with open(file_arch, 'w') as f:
            f.write(json_model)
            f.close()
        self.vae.save_weights(file_weights)

    def transform(self, x):
        return self.vae.predict(x)[0]

    def predict_original(self, x):
        return self.vae.predict(x)[1]

    def transform_predict(self, x):
        return self.vae.predict(x)[2]


class VaeSymmetryFinderConv(object):
    """Variational Autoencoder designed to find model's symmetries
    """
    def __init__(self, predict_fn, input_shape=(28, 28), output_shape=(10, ), rgb_filters=3,
                 kernel_size=3, filters=32, intermediate_dim=16, latent_dim=2, strides=2, nb_conv_layers=2,
                 intermediate_activation='relu', output_activation='sigmoid', opt='Adam', lr=0.001):
        self.predict_fn = predict_fn
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.intermediate_dim = intermediate_dim
        self.kernel_size = kernel_size
        self.filters = filters
        self.rgb_filters = rgb_filters
        self.strides = strides
        self.nb_conv_layers = nb_conv_layers
        self.opt = opt
        self.lr = lr
        self.intermediate_activation = intermediate_activation
        self.output_activation = output_activation
        self.latent_dim = latent_dim

        # It works for keras models only for now
        if isinstance(self.predict_fn, tf.keras.models.Model) or isinstance(self.predict_fn, keras.models.Model):
            for layer in self.predict_fn.layers:
                layer.trainable = False
        else:
            raise NotImplementedError

        self.inputs = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')

        for i in range(self.nb_conv_layers):
            self.filters *= 2
            self.x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                            activation='relu', strides=self.strides, padding='same')(self.inputs)
            self.x = tf.keras.layers.Dropout(0.25)(self.x)

        # shape info needed to build decoder model
        shape = K.int_shape(self.x)

        # generate latent vector Q(z|X)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.x = tf.keras.layers.Dense(self.intermediate_dim, activation=self.intermediate_activation)(self.x)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(self.x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(self.x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = tf.keras.layers.Lambda(sampling_gaussian,
                                        output_shape=(self.latent_dim,),
                                        name='z')([z_mean, z_log_var])

        self.x = tf.keras.layers.Dense(self.intermediate_dim, activation=self.intermediate_activation)(self.z)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)
        self.x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation=self.intermediate_activation)(self.x)
        self.x = tf.keras.layers.Dropout(0.25)(self.x)
        self.x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(self.x)

        for i in range(self.nb_conv_layers):
            self.x = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size,
                                                     activation='relu', strides=self.strides, padding='same')(self.x)
            self.x = tf.keras.layers.Dropout(0.25)(self.x)
            self.filters //= 2
            self.strides //= 2

        self.vae_outputs = tf.keras.layers.Conv2DTranspose(filters=self.rgb_filters,
                                                           kernel_size=self.kernel_size,
                                                           activation=self.output_activation,
                                                           padding='same',
                                                           name='decoder_output')(self.x)


        # instantiate decoder model

        self.model_output_trans = self.predict_fn(self.vae_outputs)
        self.model_output_orig = self.predict_fn(self.inputs)

        self.vae = tf.keras.models.Model(self.inputs, [self.vae_outputs, self.model_output_orig,
                                                       self.model_output_trans], name='vae_mlp')

        self.loss = tf.keras.losses.kullback_leibler_divergence(self.model_output_orig, self.model_output_trans)
        self.vae_loss = K.mean(self.loss)
        self.vae.add_loss(self.vae_loss)
        if self.opt == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        elif self.opt == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        self.vae.compile(optimizer=self.optimizer)
        print('Vae')
        self.vae.summary()

    def fit(self, X_train, x_test=None, epochs=2, batch_size=128):
        if x_test is not None:
            self.vae.fit(X_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_test, None))
        else:
            self.vae.fit(X_train,
                         epochs=epochs,
                         batch_size=batch_size)

    def save(self, save_dir=''):
        file_arch = os.path.join(save_dir, 'vae_arch.json')
        file_weights = os.path.join(save_dir, 'vae_weights.h5')
        json_model = self.vae.to_json()
        with open(file_arch, 'w') as f:
            f.write(json_model)
            f.close()
        self.vae.save_weights(file_weights)

    def transform(self, x):
        return self.vae.predict(x)[0]

    def predict_original(self, x):
        return self.vae.predict(x)[1]

    def transform_predict(self, x):
        return self.vae.predict(x)[2]


class VaeSymmetryFinderNlp(object):
    """Autoencoder using tf1 for nlp.
    ================================
    ***NOT WORKING***
    ===============================

    """
    def __init__(self, predict_fn, input_shape=(100, ), intermediate_dim=50, output_shape=(2, ),
                 wdict_len=5000, input_dtype="int32"):
        self.predict_fn = predict_fn
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.output_shape = output_shape
        if len(self.input_shape) > 1:
            self.input_dim = reduce(lambda v, w: v * w, input_shape)
        else:
            self.input_dim = self.input_shape[0]
        self.intermediate_dim = intermediate_dim
        self.wdict_len = wdict_len
        self._is_fit = False

        # It works for keras models only for now
        if isinstance(self.predict_fn, tf.keras.models.Model) or isinstance(self.predict_fn, keras.models.Model):
            for layer in self.predict_fn.layers:
                layer.trainable = False
        else:
            raise NotImplementedError

        self.sess = tf.Session()
        with self.sess.as_default():
            self.inputs = tf.keras.layers.Input(shape=self.input_shape, name='input_orig')
            self.x = tf.keras.layers.Dense(self.intermediate_dim, activation='relu', name='hidden_1')(self.inputs)
            self.p = tf.keras.layers.Dense(self.wdict_len, activation='softmax', name='p')(self.x)

            # Sample from words' probability distribution
            self.vae_outputs = tf.keras.layers.Lambda(sampling_nlp, output_shape=(self.input_dim,),
                                                      name='sampling')([self.p, self.input_dim])

            self.model_output_trans = self.predict_fn(self.vae_outputs)
            self.model_output_orig = self.predict_fn(self.inputs)

            self.loss = tf.keras.losses.kullback_leibler_divergence(self.model_output_orig, self.model_output_trans)
            self.vae_loss = K.mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer()
            self.grads = self.optimizer.compute_gradients(self.vae_loss)
            self.train_op = self.optimizer.apply_gradients(self.grads)

            tf.global_variables_initializer().run()
            print('Graph builded!')

    def fit(self, X_train, x_test=None, epochs=2, batch_size=128):

        self.losses = []
        self.test_losses = []
        nb_batches = np.floor(len(X_train) / batch_size).astype(int) + 1

        np.random.shuffle(X_train)

        for i in range(epochs):
            print('Epoch {}'.format(i))
            for j in range(nb_batches):
                batch = X_train[j * batch_size: (j * batch_size) + batch_size]
                _, loss = self.sess.run([self.train_op, self.vae_loss], feed_dict={self.inputs: batch})

                if j % 10 == 0:
                    print('Batch {} of {}'.format(j, nb_batches))
                    test_loss = self.sess.run(self.vae_loss, feed_dict={self.inputs: x_test})
                    print('Train loss = {}; Test loss = {}'.format(loss, test_loss))
            test_loss = self.sess.run(self.vae_loss, feed_dict={self.inputs: x_test})

            print('Train loss = {}; Test loss = {}'.format(loss, test_loss))
            print('+++++++++++++++++++++++++')

            self.losses.append(loss)
            self.test_losses.append(test_loss)

        print('Training done!')
        self._is_fit = True

    def save(self, save_dir=''):
        pass

    def transform(self, x):
        #assert self._is_fit
        return self.sess.run(self.vae_outputs, feed_dict={self.inputs: x})

    def predict_original(self, x):
        #assert self._is_fit
        return self.sess.run(self.model_output_orig, feed_dict={self.inputs: x})

    def transform_predict(self, x):
        #assert self._is_fit
        return self.sess.run(self.model_output_trans, feed_dict={self.inputs: x})


