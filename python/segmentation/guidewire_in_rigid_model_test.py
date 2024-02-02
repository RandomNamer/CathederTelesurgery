from guidewire_in_rigid_model import segment_guidewire, composite
import cv2
from common_configs import *
import numpy as np

image = cv2.imread(f"{PROJECT_ROOT}/Data/Datasets/RigidModelVideo-11-21/11-21-1-clip2/frame1069.jpg")
s, e, i = segment_guidewire(image, connnected_components_thresholding='otsu', allow_fragmentation=False)
print(np.unique(s, return_counts=True))


cv2.imwrite("./test.jpg", composite(image, s, e, alpha=0.5, speed=114.514))
print(i)
print(e)