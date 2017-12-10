import numpy as np
from scipy import io
import os
import tensorflow as tf
from fnmatch import fnmatch
from lib.binvox_rw import read_as_3d_array

voxels = io.loadmat("test.mat", squeeze_me=True)["mydata"]
voxels = np.asarray(voxels)

print(voxels.shape)
