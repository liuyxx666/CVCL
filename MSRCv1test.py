import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path="datasets/"
mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
X_data = mat['X']
num_views = X_data.shape[1]
print(num_views)
print(X_data)
print(X_data.shape)
data_views = list()
for idx in range(num_views):
    data_views.append(X_data[0, idx].astype(np.float32))
print(data_views)
print(data_views[0])
scaler = MinMaxScaler()
for idx in range(num_views):
    data_views[idx] = scaler.fit_transform(data_views[idx])
print(data_views)
labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
print(labels)
#print(mat['Y'])