import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt
from string import ascii_lowercase


mat_file = loadmat("datasets/notMNIST_small.mat")
L = len(mat_file['labels'])

mat_file['images'] = np.array([mat_file['images'][:,:,i] for i in range(L)])
vf = np.vectorize(lambda x: ascii_lowercase[int(x)])
mat_file['labels'] = vf(mat_file['labels'])

test_len = 18724//4
train_len = 18724 - test_len
train_X = mat_file['images'][:train_len]
train_y = mat_file['labels'][:train_len]
test_X = mat_file['images'][test_len:]
test_y = mat_file['labels'][test_len:]


def plot(i=1):
    print("Label: {}".format(train_y[i]))
    plt.imshow(train_X[i])
    plt.show()