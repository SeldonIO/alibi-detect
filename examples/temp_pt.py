import torch
import torchvision
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from alibi_detect.cd import NMEDrift

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mnist_train_ds = torchvision.datasets.MNIST('/home/oliver/Projects/alibi-detect/examples', train=True)
mnist_test_ds = torchvision.datasets.MNIST('/home/oliver/Projects/alibi-detect/examples', train=False)

train_x, train_y = mnist_train_ds.train_data, mnist_train_ds.train_labels
test_x, test_y = mnist_test_ds.test_data, mnist_test_ds.test_labels

x_ref = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)
x_ref[:,0,0,0] = 10* np.ones((1000,)).astype(np.float32)
x_ref[:,0,0,1] = 10* np.ones((1000,)).astype(np.float32)
x_ref[:,0,1,1] = 10* np.ones((1000,)).astype(np.float32)
x_ref[:,0,1,0] = 10* np.ones((1000,)).astype(np.float32)
x = np.random.normal(size=(1000, 1, 28, 28)).astype(np.float32)

# data = train_x[:,None ,: , :].numpy().astype(np.float32)/255.
# perm = np.random.permutation(np.arange(data.shape[0]))
# x_ref = data[perm][:10000]
# x_ref = x_ref[train_y[perm][:10000]!=0]
# x = data[perm][10000:20000]


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
    nn.ReLU(),
    nn.Flatten()
).to(device)

for param in conv_encoder.parameters():
    if len(param.shape) > 1:
        torch.nn.init.xavier_uniform(param)
    # param.requires_grad = False
kernel = DeepKernel(conv_encoder, eps=0.01)
# kernel = DeepKernel(nn.Flatten(), eps=0.01)
# kernel = GaussianRBF(trainable=True)

cd = NMEDrift(
    x_ref, 
    kernel, 
    backend='pytorch', 
    verbose=1,
    learning_rate=1e-2, 
    epochs=50, 
    batch_size=32,
    device='cpu'
)

preds = cd.predict(x)
plt.imshow(preds['data']['test_locs'][0,0])
plt.colorbar()
plt.savefig('final_test_locs.png')
print('done')