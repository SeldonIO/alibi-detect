import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from alibi_detect.utils.tensorflow.kernels import DeepKernel, GaussianRBF
from alibi_detect.cd import NMEDrift

### Config
DATA_TYPE = ['mnist', 'gaussian_corner'][0]  
KERNEL_TYPE = ['deep', 'shallow'][1]
LEARNING_RATE = 1e-3
TEST_LOCS_SAVE_PATH = 'final_test_locs.png'
MNIST_PATH = '/home/oliver/Projects/alibi-detect/examples'
DOWNLOAD = False  # set to True for first run
MISSING_NUMBER = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Data
if DATA_TYPE == 'mnist':
    mnist_train_ds = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=DOWNLOAD)
    train_x, train_y = mnist_train_ds.train_data, mnist_train_ds.train_labels

    data = train_x[:, None, : , :].numpy().astype(np.float32)/255.
    perm = np.random.permutation(np.arange(data.shape[0]))
    x_ref = data[perm][:10000]
    x = data[perm][10000:20000]
    x_ref = x_ref[train_y[perm][:10000] != MISSING_NUMBER]  # no MISSING_NUMBER in ref
else:
    x_ref = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)
    x = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)
    x_ref[:,0,:2,:2] = 10* np.ones((1000,2,2)).astype(np.float32)  # ref has highlighted corner

x_ref = x_ref.transpose((0, 2, 3, 1))
x = x.transpose((0, 2, 3, 1))

### Kernel
if KERNEL_TYPE == 'deep':
    # Maps (1, 28, 28) onto (20, 1)
    conv_encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 3, input_shape=(28, 28, 1), padding='same'),  
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(2), # 14 x 14
        tf.keras.layers.Conv2D(12, 3, padding='valid'),  
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(2), # 6 x 6
        tf.keras.layers.Conv2D(16, 3, padding='same'),  
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(2), # 3 x 3
        tf.keras.layers.Conv2D(20, 3, padding='valid'),  
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten()
        ]
    )
    # for param in conv_encoder.parameters():
    #     param.requires_grad = False
    kernel = DeepKernel(conv_encoder, eps=0.01)
else:
    kernel = GaussianRBF(trainable=True)

### Detector
cd = NMEDrift(
    x_ref, 
    kernel, 
    backend='tensorflow', 
    verbose=1,
    learning_rate=LEARNING_RATE, 
    epochs=20, 
    batch_size=32
)

### Preds and results
preds = cd.predict(x)
plt.imshow(preds['data']['test_locs'][0,:,:,0])
plt.colorbar()
plt.savefig(TEST_LOCS_SAVE_PATH)
print('done')

###
# Try this with and without dividing output of alibi_detect.utils.tensorflow.kernels.GaussianRBF by 100
# or equivalently use kernel DeepKernel(lambda x: 0, eps=0.01)