from functools import partial
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple


def trainer(
        model: tf.keras.Model,
        loss_fn: tf.keras.losses,
        x_train: np.ndarray,
        y_train: np.ndarray = None,
        dataset: tf.keras.utils.Sequence = None,
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss_fn_kwargs: dict = None,
        preprocess_fn: Callable = None,
        epochs: int = 20,
        reg_loss_fn: Callable = (lambda model: 0),
        batch_size: int = 64,
        buffer_size: int = 1024,
        verbose: bool = True,
        log_metric:  Tuple[str, "tf.keras.metrics"] = None,
        callbacks: tf.keras.callbacks = None
) -> None:
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    x_train
        Training data.
    y_train
        Training labels.
    dataset
        Training dataset.
    optimizer
        Optimizer used for training.
    loss_fn_kwargs
        Kwargs for loss function.
    preprocess_fn
        Preprocessing function applied to each training batch.
    epochs
        Number of training epochs.
    reg_loss_fn
        Allows an additional regularisation term to be defined as reg_loss_fn(model)
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    log_metric
        Additional metrics whose progress will be displayed if verbose equals True.
    callbacks
        Callbacks used during training.
    """
    if not isinstance(dataset, tf.keras.utils.Sequence):  # create dataset
        train_data = x_train if y_train is None else (x_train, y_train)
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    n_minibatch = len(dataset)

    if loss_fn_kwargs:
        loss_fn = partial(loss_fn, **loss_fn_kwargs)

    # iterate over epochs
    for epoch in range(epochs):
        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)
        if hasattr(dataset, 'on_epoch_end'):
            dataset.on_epoch_end()
        loss_val_ma = 0.
        for step, data in enumerate(dataset):
            x, y = data if len(data) == 2 else (data, None)
            if isinstance(preprocess_fn, Callable):  # type: ignore
                x = preprocess_fn(x)
            with tf.GradientTape() as tape:
                y_hat = model(x)
                y = x if y is None else y
                if isinstance(loss_fn, Callable):  # type: ignore
                    args = [y, y_hat] if tf.is_tensor(y_hat) else [y] + list(y_hat)
                    loss = loss_fn(*args)
                else:
                    loss = 0.
                if model.losses:  # additional model losses
                    loss += sum(model.losses)
                loss += reg_loss_fn(model)  # alternative way they might be specified

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if verbose:
                loss_val = loss.numpy()
                if loss_val.shape:
                    if loss_val.shape[0] != batch_size:
                        if len(loss_val.shape) == 1:
                            shape = (batch_size - loss_val.shape[0], )
                        elif len(loss_val.shape) == 2:
                            shape = (batch_size - loss_val.shape[0], loss_val.shape[1])  # type: ignore
                        add_mean = np.ones(shape) * loss_val.mean()
                        loss_val = np.r_[loss_val, add_mean]
                loss_val_ma = loss_val_ma + (loss_val - loss_val_ma) / (step + 1)
                pbar_values = [('loss_ma', loss_val_ma)]
                if log_metric is not None:
                    log_metric[1](y, y_hat)
                    pbar_values.append((log_metric[0], log_metric[1].result().numpy()))
                pbar.add(1, values=pbar_values)
