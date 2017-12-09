import numpy as np
from scipy import io

voxels = io.loadmat('test.mat', squeeze_me=True)['mydata']
voxels = np.asarray(voxels)


print(voxels.shape)


# io.savemat('test.mat', {'mydata': voxels})

