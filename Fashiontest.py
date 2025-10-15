import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path="datasets/"
data_views = list()
mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
print( mat)
X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
print(mat['X1'])
print(mat['X1'].shape)
print(X1)
print(X1.shape)
data_views.append(X1)
data_views.append(X2)
data_views.append(X3)
print(data_views)