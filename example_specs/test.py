from alibi_detect.factory import DriftBuilder
import numpy as np

x_ref = np.random.normal(size=100)
detector = DriftBuilder(x_ref, 'mmd_image.yaml')
print(detector)
