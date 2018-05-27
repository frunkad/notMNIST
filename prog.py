import numpy as np 
from scipy.io import loadmat
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from string import ascii_lowercase
from random import randint
tf.logging.set_verbosity(tf.logging.INFO)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

mat_file = loadmat("datasets/notMNIST_large.mat")
L = len(mat_file['labels'][0]) # For short, use mat_file['labels'], for large use mat_file['labels'][0]
print("Length of the Dataset: ",L)
mat_file['images'] = np.array([mat_file['images'][:,:,i] for i in range(L)])
# vf = np.vectorize(lambda x: ascii_lowercase[int(x)])
vf = np.vectorize(lambda x: int(x))
mat_file['labels'] = vf(mat_file['labels'][0])
possible_values = set(mat_file['labels'])
print("Shape of Images: {}\nShape of Labels: {}\n"+bcolors.BOLD+"Possible Values: {}".format(mat_file['images'].shape,mat_file['labels'].shape,possible_values)+bcolors.ENDC)
seed = randint(1,1000)
np.random.seed(seed)
np.random.shuffle(mat_file['images'])
np.random.seed(seed)
np.random.shuffle(mat_file['labels'])

test_len = L//4
train_len = L - test_len
trainX = mat_file['images'][test_len:]
trainy = mat_file['labels'][test_len:]
testX = mat_file['images'][:test_len]
testy = mat_file['labels'][:test_len]


def plot(i=1):
    print("Label: {}".format(trainy[i]))
    plt.imshow(trainX[i].reshape(28,28))
    plt.show()

#START
trainX = np.reshape(trainX,(train_len,28*28))
testX = np.reshape(testX,(test_len,28*28))

trainX /= 255
testX /= 255

trainy = tf.keras.utils.to_categorical(trainy,10)
testy = tf.keras.utils.to_categorical(testy,10)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.elu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
model.fit(trainX,trainy,epochs=5)
loss,accuracy = model.evaluate(testX,testy)
print(accuracy)