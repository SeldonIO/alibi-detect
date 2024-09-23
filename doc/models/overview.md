# Overview

Models and/or building blocks that can be useful outside of outlier, adversarial or drift detection can be found under `alibi_detect.models`. Main implementations:

- [PixelCNN++](https://arxiv.org/abs/1701.05517): `from alibi_detect.models.tensorflow import PixelCNN`

- Variational Autoencoder: `from alibi_detect.models.tensorflow import VAE`

- Sequence-to-sequence model: `from alibi_detect.models.tensorflow import Seq2Seq`

- ResNet: `from alibi_detect.models.tensorflow import resnet`

Pre-trained TensorFlow ResNet-20/32/44 models on CIFAR-10 can be found on our [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/seldon-models/alibi-detect/classifier/cifar10/?organizationId=156002945562&project=seldon-pub) and can be fetched as follows:

```python
from alibi_detect.utils.fetching import fetch_tf_model

model = fetch_tf_model('cifar10', 'resnet32')
```

