import torch
import torchvision
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from alibi_detect.utils.tensorflow.kernels import DeepKernel, GaussianRBF
from alibi_detect.cd import NMEDrift

# x = tf.random.normal((32,28,28,1))
# model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, input_shape=(28,28,1))])
# out = model(x)

mnist_train_ds = torchvision.datasets.MNIST('examples/', train=True, download=True)
mnist_test_ds = torchvision.datasets.MNIST('examples/', train=False, download=True)

train_x, train_y = mnist_train_ds.train_data, mnist_train_ds.train_labels
test_x, test_y = mnist_test_ds.test_data, mnist_test_ds.test_labels

# x_ref = train_x[:,None ,: , :].numpy().astype(np.float32)/255.
# # x_ref = np.ones_like(x_ref)
# x = x_ref[40000:]
# x_ref = x_ref[:40000]
# # x = test_x.numpy().astype(np.float32)[:,None ,: , :]/255.

# x_ref = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)
# x_ref[:,0,0,0] = 4* np.ones((1000,)).astype(np.float32)
# x_ref[:,0,0,1] = 4* np.ones((1000,)).astype(np.float32)
# x_ref[:,0,1,0] = 4* np.ones((1000,)).astype(np.float32)
# x_ref[:,0,1,1] = 4* np.ones((1000,)).astype(np.float32)
# x = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)

data = train_x[:,None ,: , :].numpy().astype(np.float32)/255.
perm = np.random.permutation(np.arange(data.shape[0]))
x_ref = data[perm][:10000]
x_ref = x_ref[train_y[perm][:10000]!=0]
x = data[perm][10000:20000]

x_ref = x_ref.transpose((0, 2, 3, 1))
x = x.transpose((0, 2, 3, 1))

initter = tf.keras.initializers.RandomUniform(-1/9, 1/9)

conv_encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, input_shape=(28, 28, 1), padding='same', 
        kernel_initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(9*1), 1/np.sqrt(9*1))),  
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(2),                             # 14 x 14
    tf.keras.layers.Conv2D(12, 3, padding='valid', 
        kernel_initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(9*8), 1/np.sqrt(9*8))),  
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(2),                              # 6 x 6
    tf.keras.layers.Conv2D(16, 3, padding='same', 
        kernel_initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(9*12), 1/np.sqrt(9*12))),  
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(2),                              # 3 x 3
    tf.keras.layers.Conv2D(20, 3, padding='valid', 
        kernel_initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(9*16), 1/np.sqrt(9*16))),  
    tf.keras.layers.ReLU(),
    tf.keras.layers.Flatten()
    ]
)
# for param in conv_encoder.parameters():
#     param.requires_grad = False
kernel = DeepKernel(conv_encoder, eps=0.01)
# kernel = DeepKernel(tf.keras.layers.Flatten(), eps=0.01)
# kernel = GaussianRBF(trainable=True)

cd = NMEDrift(
    x_ref, 
    kernel, 
    backend='tensorflow', 
    verbose=1,
    learning_rate=1e-2, 
    epochs=50, 
    batch_size=32
)

preds = cd.predict(x)
plt.imshow(preds['data']['test_locs'][0,:,:,0])
plt.colorbar()
plt.savefig('final_test_locs.png')
print('done')