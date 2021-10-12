from alibi_detect.factory import DetectorFactory
import numpy as np

x_ref = np.random.normal(size=100)
detector = DetectorFactory(x_ref, 'mmd_image.yaml')
print(detector)
