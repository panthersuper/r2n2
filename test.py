import numpy as np
from scipy import io
import os
import tensorflow as tf
from fnmatch import fnmatch
from lib.binvox_rw import read_as_3d_array


def addBoundary(vox):
    newvox = vox

    filter_np = np.array(
            [
                [0,0,0],
                [0,1,0],
                [0,0,0],
                [0,1,0],
                [1,0,1],
                [0,1,0],
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ])
    filter_np = np.reshape(filter_np,[3,3,3,1,1])
    vox_5d = np.reshape(vox,[1,1,32,32,32])
    vox_5d = np.moveaxis(vox_5d, 1, 4)

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    output = tf.nn.conv3d(vox_5d,filter_np,padding="SAME",strides=[1,1,1,1,1])

    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    sess = tf.Session(config=config)

    with sess.as_default():

        # print(type(tf.constant([1,2,3]).eval()))
        output = np.reshape(output.eval(),[32,32,32])

    neighbor_has_0 = output < 6
    i_am_1 = vox
    i_am_boundary = np.logical_and(neighbor_has_0,i_am_1)*1

    newvox[(i_am_boundary==1)] = 2

    # unique, counts = np.unique(newvox, return_counts=True)
    # print(dict(zip(unique, counts)),"newvox")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    return vox


vox_root = './ShapeNet/ShapeNetVox32'


for path, subdirs, files in os.walk(vox_root):
    for name in files:
        if fnmatch(name, '*.binvox'):


            mypath = os.path.join(path,name)     
            savingpath = os.path.join(path,"model.mat")  
            with open(mypath, 'rb') as f:

                voxel = read_as_3d_array(f)
                voxel = np.array(voxel.data).astype(np.float32)
                voxel = addBoundary(voxel)

                unique, counts = np.unique(voxel, return_counts=True)
                print(dict(zip(unique, counts)),"voxel")

                io.savemat(savingpath, {'mydata': voxel})
