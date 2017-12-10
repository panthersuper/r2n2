import numpy as np
from scipy import io
import os
import tensorflow as tf
from fnmatch import fnmatch
from lib.binvox_rw import read_as_3d_array

vox_root = './ShapeNet/ShapeNetVox32'

count= 0
for path, subdirs, files in os.walk(vox_root):
    for name in files:
        if fnmatch(name, '*.mat'):
            count+=1

print(count)

count= 0
for path, subdirs, files in os.walk(vox_root):
    for name in files:
        if fnmatch(name, '*.binvox'):
            count+=1

print(count)