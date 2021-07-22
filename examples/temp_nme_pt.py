import torch
import torchvision
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import wandb

from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from alibi_detect.cd import NMEDrift

### Config
DATA_TYPE = ['mnist', 'gaussian_corner'][0]  
KERNEL_TYPE = ['deep', 'shallow'][1]
LEARNING_RATE = 1e-3
TEST_LOCS_SAVE_PATH = 'final_test_locs.png'
MNIST_PATH = '/home/oliver/Projects/alibi-detect/examples'
DOWNLOAD = False  # set to True for first run
MISSING_NUMBER = 0

run = wandb.init(project='nme')
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

### Kernel
if KERNEL_TYPE == 'deep':
    # Maps (1, 28, 28) onto (20, 1)
    conv_encoder = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2),                             # 14 x 14
        nn.Conv2d(8, 12, kernel_size=3, padding=0),  
        nn.ReLU(),
        nn.MaxPool2d(2),                              # 6 x 6
        nn.Conv2d(12, 16, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2),                              # 3 x 3
        nn.Conv2d(16, 20, kernel_size=3, padding=0),  
        nn.Flatten()
    ).to(device)
    # for param in conv_encoder.parameters():
        # param.requires_grad = False
    kernel = DeepKernel(conv_encoder, eps=0.01)
else:
    kernel = GaussianRBF(trainable=True)

### Detector
cd = NMEDrift(
    x_ref, 
    kernel, 
    backend='pytorch', 
    verbose=1,
    learning_rate=LEARNING_RATE, 
    epochs=20, 
    batch_size=32
)
wandb.watch(cd._detector.nme_embedder, log='all', log_freq=1)

### Preds and results
preds = cd.predict(x)
plt.imshow(preds['data']['test_locs'][0,0])
plt.colorbar()
plt.savefig(TEST_LOCS_SAVE_PATH)
print('done')

###
# Try this with and without dividing output of alibi_detect.utils.pytorch.kernels.GaussianRBF by 100
# or equivalently use kernel DeepKernel(lambda x: 0, eps=0.01)