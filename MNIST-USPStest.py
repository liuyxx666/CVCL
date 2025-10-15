import os
import scipy.io as sio
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

path="datasets/"
data_views = list()
mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
X1 = mat['X1'].astype(np.float32)
X2 = mat['X2'].astype(np.float32)
print(X1)
print(X2)
print(X1.shape)
data_views.append(X1.reshape(X1.shape[0], -1))
data_views.append(X2.reshape(X2.shape[0], -1))
print(data_views)
num_views = len(data_views)
labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
print(labels)
print(mat)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for idx in range(num_views):
    data_views[idx] = torch.from_numpy(data_views[idx]).to(device)
print(data_views)