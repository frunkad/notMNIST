import numpy as np 
from scipy.io import loadmat
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from string import ascii_lowercase
from random import randint
import sys,time,os,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpuoff",help="Turn the GPU Device off",action="store_true")
parser.add_argument("-d","--dataset",choices=["small","large"],help="small dataset or large",default="small")
parser.add_argument("-s","--save",help="save to hdf5 file",default = False)
argv = parser.parse_args()

if argv.gpuoff:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.logging.set_verbosity(tf.logging.ERROR)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

DATASET = "small" # or "large"

if argv.dataset == "large":
    DATASET = "large"




mat_file = loadmat("datasets/notMNIST_{}.mat".format(DATASET))

L = len(mat_file['labels'][0]) if DATASET == "large" else len(mat_file['labels']) # For short, use mat_file['labels'], for large use mat_file['labels'][0]
mat_file['images'] = np.array([mat_file['images'][:,:,i] for i in range(L)])
# vf = np.vectorize(lambda x: ascii_lowercase[int(x)])
vf = np.vectorize(lambda x: int(x))
mat_file['labels'] = vf(mat_file['labels'][0]) if DATASET == "large" else vf(mat_file['labels'])
possible_values = set(mat_file['labels'])

print(bcolors.BOLD+"Length of the Dataset: {}\nShape of Images: {}\nShape of Labels: {}\nPossible Values: {}".format(L,mat_file['images'].shape,mat_file['labels'].shape,possible_values)+bcolors.ENDC)


seed = randint(1,1000)
np.random.seed(seed)
np.random.shuffle(mat_file['images'])
np.random.seed(seed)
np.random.shuffle(mat_file['labels'])

test_len = L//5
train_len = L - test_len
trainX = mat_file['images'][test_len:]
trainy = mat_file['labels'][test_len:]
testX = mat_file['images'][:test_len]
testy = mat_file['labels'][:test_len]

print(bcolors.BOLD+"Test Data: {}\nTrain Data: {}".format(test_len,train_len)+bcolors.ENDC)

def plot(i=1):
    print("Label: {}".format(trainy[i]))
    plt.imshow(trainX[i].reshape(28,28))
    plt.show()

# #START
trainX = np.reshape(trainX,(train_len,28*28))
testX = np.reshape(testX,(test_len,28*28))

trainX /= 255
testX /= 255

trainy = tf.keras.utils.to_categorical(trainy,10)
testy = tf.keras.utils.to_categorical(testy,10)

trainX = np.expand_dims(trainX,-1)
testX = np.expand_dims(trainX,-1)

def tostr(s):
    if callable(s):
        return s.__name__
    else:
        return str(s)

def create_networks():
    trainX = np.reshape(trainX,(train_len,28*28))
    testX = np.reshape(testX,(test_len,28*28))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.65))
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
    # 
    # We will now compile and print out a summary of our model
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    return model

def try_model(
                epochs = 5,
                save = False,
                batch_size = None,
                model = create_networks()
            ):
    try_model.count += 1


    
    model.fit(trainX,trainy,epochs=epochs,verbose=1,batch_size=batch_size)
    lossed,accuracy = model.evaluate(testX,testy,verbose=1)
    print("====================================\nModel ",try_model.count," : ")
    # print("\tActivator 1: {}\n\tActivator 2: {}\n\tLoss: {}\n\tOptimizer: {}\n\tEpochs: {}".format(tostr(activator1),tostr(activator2),loss,tostr(optimizer),epochs))
    model.summary()
    print(bcolors.OKGREEN+"\tAccuracy: {}".format(accuracy)+bcolors.ENDC)
    print(bcolors.WARNING+"\tLoss: {}".format(lossed)+bcolors.ENDC)
    if save:
        tf.keras.models.save_model(model,"model-"+DATASET+"-"+time.ctime()+".hdf5")

try_model.count = 0

# losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'cosine_proximity', 'hinge', 'kullback_leibler_divergence', 'logcosh', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error', 'poisson', 'sparse_categorical_crossentropy', 'squared_hinge']
# for loss in losses:
#     print("Loss: "+loss)
#     try_model(loss = loss)


if __name__ == "__main__":
    try_model(save=argv.save)
