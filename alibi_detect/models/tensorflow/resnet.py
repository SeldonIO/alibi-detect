# implementation adopted from https://github.com/tensorflow/models
# TODO: proper train-val-test split
import argparse
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     Dense, Input, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from typing import Callable, Tuple, Union

# parameters specific for CIFAR-10 training
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-4
LR_SCHEDULE = [(0.1, 91), (0.01, 136), (0.001, 182)]  # (multiplier, epoch to start) tuples
BASE_LEARNING_RATE = 0.1
HEIGHT, WIDTH, NUM_CHANNELS = 32, 32, 3


def l2_regulariser(l2_regularisation: bool = True):
    """
    Apply L2 regularisation to kernel.

    Parameters
    ----------
    l2_regularisation
        Whether to apply L2 regularisation.

    Returns
    -------
    Kernel regularisation.
    """
    return l2(L2_WEIGHT_DECAY) if l2_regularisation else None


def identity_block(x_in: tf.Tensor,
                   filters: Tuple[int, int],
                   kernel_size: Union[int, list, Tuple[int]],
                   stage: int,
                   block: str,
                   l2_regularisation: bool = True) -> tf.Tensor:
    """
    Identity block in ResNet.

    Parameters
    ----------
    x_in
        Input Tensor.
    filters
        Number of filters for each of the 2 conv layers.
    kernel_size
        Kernel size for the conv layers.
    stage
        Stage of the block in the ResNet.
    block
        Block within a stage in the ResNet.
    l2_regularisation
        Whether to apply L2 regularisation.

    Returns
    -------
    Output Tensor of the identity block.
    """
    # name of block
    conv_name_base = 'res' + str(stage) + '_' + block + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + block + '_branch'

    filters_1, filters_2 = filters
    bn_axis = 3  # channels last format

    x = Conv2D(
        filters_1,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name=conv_name_base + '2a')(x_in)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters_2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2b')(x)

    x = Add()([x, x_in])
    x = Activation('relu')(x)
    return x


def conv_block(x_in: tf.Tensor,
               filters: Tuple[int, int],
               kernel_size: Union[int, list, Tuple[int]],
               stage: int,
               block: str,
               strides: Tuple[int, int] = (2, 2),
               l2_regularisation: bool = True) -> tf.Tensor:
    """
    Conv block in ResNet with a parameterised skip connection to reduce the width and height
    controlled by the strides.

    Parameters
    ----------
    x_in
        Input Tensor.
    filters
        Number of filters for each of the 2 conv layers.
    kernel_size
        Kernel size for the conv layers.
    stage
        Stage of the block in the ResNet.
    block
        Block within a stage in the ResNet.
    strides
        Stride size applied to reduce the image size.
    l2_regularisation
        Whether to apply L2 regularisation.

    Returns
    -------
    Output Tensor of the conv block.
    """
    # name of block
    conv_name_base = 'res' + str(stage) + '_' + block + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + block + '_branch'

    filters_1, filters_2 = filters
    bn_axis = 3  # channels last format

    x = Conv2D(
        filters_1,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name=conv_name_base + '2a')(x_in)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters_2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2b')(x)

    shortcut = Conv2D(
        filters_2,
        (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name=conv_name_base + '1')(x_in)
    shortcut = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_block(x_in: tf.Tensor,
                 size: int,
                 filters: Tuple[int, int],
                 kernel_size: Union[int, list, Tuple[int]],
                 stage: int,
                 strides: Tuple[int, int] = (2, 2),
                 l2_regularisation: bool = True) -> tf.Tensor:
    """
    Block in ResNet combining a conv block with identity blocks.

    Parameters
    ----------
    x_in
        Input Tensor.
    size
        The ResNet block consists of 1 conv block and size-1 identity blocks.
    filters
        Number of filters for each of the conv layers.
    kernel_size
        Kernel size for the conv layers.
    stage
        Stage of the block in the ResNet.
    strides
        Stride size applied to reduce the image size.
    l2_regularisation
        Whether to apply L2 regularisation.

    Returns
    -------
    Output Tensor of the conv block.
    """
    x = conv_block(
        x_in,
        filters,
        kernel_size,
        stage,
        'block0',
        strides=strides,
        l2_regularisation=l2_regularisation
    )

    for i in range(size - 1):
        x = identity_block(
            x,
            filters,
            kernel_size,
            stage,
            f'block{i + 1}',
            l2_regularisation=l2_regularisation
        )

    return x


def resnet(num_blocks: int,
           classes: int = 10,
           input_shape: Tuple[int, int, int] = (32, 32, 3)) -> tf.keras.Model:
    """
    Define ResNet.

    Parameters
    ----------
    num_blocks
        Number of ResNet blocks.
    classes
        Number of classification classes.
    input_shape
        Input shape of an image.

    Returns
    -------
    ResNet as a tf.keras.Model.
    """
    bn_axis = 3  # channels last format
    l2_regularisation = True

    x_in = Input(shape=input_shape)
    x = ZeroPadding2D(
        padding=(1, 1),
        name='conv1_pad')(x_in)
    x = Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regulariser(l2_regularisation),
        name='conv1')(x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = resnet_block(
        x_in=x,
        size=num_blocks,
        filters=(16, 16),
        kernel_size=3,
        stage=2,
        strides=(1, 1),
        l2_regularisation=True
    )

    x = resnet_block(
        x_in=x,
        size=num_blocks,
        filters=(32, 32),
        kernel_size=3,
        stage=3,
        strides=(2, 2),
        l2_regularisation=True
    )

    x = resnet_block(
        x_in=x,
        size=num_blocks,
        filters=(64, 64),
        kernel_size=3,
        stage=4,
        strides=(2, 2),
        l2_regularisation=True
    )

    x = tf.reduce_mean(x, axis=(1, 2))  # take mean across width and height
    x_out = Dense(
        classes,
        activation='softmax',
        kernel_initializer=RandomNormal(stddev=.01),
        kernel_regularizer=l2(L2_WEIGHT_DECAY),
        bias_regularizer=l2(L2_WEIGHT_DECAY),
        name='fc10')(x)

    model = Model(x_in, x_out, name='resnet')
    return model


def learning_rate_schedule(current_epoch: int,
                           current_batch: int,
                           batches_per_epoch: int,
                           batch_size: int) -> float:
    """
    Linear learning rate scaling and learning rate decay at specified epochs.

    Parameters
    ----------
    current_epoch
        Current training epoch.
    current_batch
        Current batch with current epoch, not used.
    batches_per_epoch
        Number of batches or steps in an epoch, not used.
    batch_size
        Batch size.

    Returns
    -------
    Adjusted learning rate.
    """
    del current_batch, batches_per_epoch  # not used
    initial_learning_rate = BASE_LEARNING_RATE * batch_size / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if current_epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    return learning_rate


class LearningRateBatchScheduler(Callback):

    def __init__(self, schedule: Callable, batch_size: int, steps_per_epoch: int):
        """
        Callback to update learning rate on every batch instead of epoch.

        Parameters
        ----------
        schedule
            Function taking the epoch and batch index as input which returns the new
            learning rate as output.
        batch_size
            Batch size.
        steps_per_epoch
            Number of batches or steps per epoch.
        """
        super(LearningRateBatchScheduler, self).__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.epochs = -1
        self.prev_lr = -1

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.epochs += 1

    def on_batch_begin(self, batch, logs=None):
        """Executes before step begins."""
        lr = self.schedule(self.epochs,
                           batch,
                           self.steps_per_epoch,
                           self.batch_size)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if lr != self.prev_lr:
            self.model.optimizer.learning_rate = lr  # lr should be a float
            self.prev_lr = lr
            tf.compat.v1.logging.debug(
                'Epoch %05d Batch %05d: LearningRateBatchScheduler '
                'change learning rate to %s.', self.epochs, batch, lr)


def preprocess_image(x: np.ndarray, is_training: bool = True) -> np.ndarray:
    if is_training:
        # resize image and add 4 pixels to each side
        x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)

        # randomly crop a [HEIGHT, WIDTH] section of the image
        x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])

        # randomly flip the image horizontally
        x = tf.image.random_flip_left_right(x)

    # standardise by image
    x = tf.image.per_image_standardization(x).numpy().astype(np.float32)
    return x


def scale_by_instance(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    xmean = x.mean(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    xstd = x.std(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    x_scaled = (x - xmean) / (xstd + eps)
    return x_scaled


def run(num_blocks: int,
        epochs: int,
        batch_size: int,
        model_dir: Union[str, os.PathLike],
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        validation_freq: int = 10,
        verbose: int = 2,
        seed: int = 1,
        serving: bool = False
        ) -> None:

    # load and preprocess CIFAR-10 data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = scale_by_instance(X_test.astype('float32'))  # can already preprocess test data
    y_train = y_train.astype('int64').reshape(-1, )
    y_test = y_test.astype('int64').reshape(-1, )

    # define and compile model
    model = resnet(num_blocks, classes=num_classes, input_shape=input_shape)
    optimizer = SGD(learning_rate=BASE_LEARNING_RATE, momentum=0.9)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['sparse_categorical_accuracy']
    )

    # set up callbacks
    steps_per_epoch = X_train.shape[0] // batch_size
    ckpt_path = Path(model_dir).joinpath('model.h5')
    callbacks = [
        ModelCheckpoint(
            ckpt_path,
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=False
        ),
        LearningRateBatchScheduler(
            schedule=learning_rate_schedule,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch
        )
    ]

    # data augmentation and preprocessing
    datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

    # train
    model.fit(
        x=datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        validation_freq=validation_freq,
        validation_data=(X_test, y_test),
        shuffle=True,
        verbose=verbose
    )

    if serving:
        tf.saved_model.save(model, model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10.")
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--validation_freq', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--serving', type=bool, default=False)
    args = parser.parse_args()
    run(
        args.num_blocks,
        args.epochs,
        args.batch_size,
        args.model_dir,
        num_classes=args.num_classes,
        validation_freq=args.validation_freq,
        verbose=args.verbose,
        seed=args.seed,
        serving=args.serving
    )
