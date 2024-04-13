import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from DL3 import *
import numpy as np


IMAGE_SIZE = (32, 128)
count = 0

# avoid too much data - if over 250 images per class, only take 250

def load_train_data(folder_path):
    global count
    X = []
    Y = []
    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip", "other"]

    for i, class_name in enumerate(class_names):
        currX = []
        currY = []
        cnt = 0
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            cnt += 1
            if (cnt == 250):
                break
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path)
            image = image.resize(IMAGE_SIZE)
            
            # Convert grayscale images to RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image_array = np.array(image).reshape(3*IMAGE_SIZE[0]*IMAGE_SIZE[1],)
            currX.append(image_array)
            currY.append(i)
            count += 1
            if count % 249 == 0:
                print(i)
        X.append(currX)
        Y.append(currY)
    
    return X, Y




X, Y = load_train_data('dataset/train')
for i in range(len(X)):
    X[i] = np.array(X[i])
    X[i] = X[i] / 255.0 - 0.5
    Y[i] = np.array(Y[i])

Xtrain, Xtest, Ytrain, Ytest = [], [], [], []
for i in range(len(X)):
    Xtrain_class, Xtest_class, Ytrain_class, Ytest_class = train_test_split(X[i], Y[i], test_size=0.3)
    # dont add the lists but the elements of the lists
    for element in Xtrain_class:
        Xtrain.append(element)
    for element in Xtest_class:
        Xtest.append(element)
    for element in Ytrain_class:
        Ytrain.append(element)
    for element in Ytest_class:
        Ytest.append(element)

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)

Xtrain = Xtrain.T
Xtest = Xtest.T

np.random.seed(42) # the meaning of life, the universe, and everything - Douglas Adams (also a great seed) 
random.seed(42) 

model = DLModel()

HiddenLayer1 = DLLayer("Hidden Layer 1", 256, (3*IMAGE_SIZE[0]*IMAGE_SIZE[1],), activation="leaky_relu", W_initialization="random", learning_rate=0.4, random_scale=0.1)
model.add(HiddenLayer1)

HiddenLayer2 = DLLayer("Hidden Layer 2", 64, (256,), activation="trim_sigmoid", W_initialization="random", learning_rate=1, random_scale=1)
model.add(HiddenLayer2)

OutputLayer = DLLayer("Output Layer", 6, (64,), activation="trim_softmax", W_initialization="random", learning_rate=0.1, random_scale=0.001)
model.add(OutputLayer)

model.compile("categorical_cross_entropy")

costs = model.train(Xtrain, Ytrain, 300)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 2s)')
plt.show()
print("train accuracy:", np.mean(model.predict(Xtrain) == Ytrain))
print("test accuracy:", np.mean(model.predict(Xtest) == Ytest))

#model.save_weights("./weights")