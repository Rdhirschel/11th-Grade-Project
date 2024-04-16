from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

IMAGE_SIZE = (32, 32) # 256x256 gives same results but takes too much time
classNames = ["tulip", "sunflower", "rose", "dandelion", "iris"]

def load_train_data(image_directory="dataset\\train"):
    X_train, X_test, Y_train, Y_test = [], [], [], []

    # common_daisy, tulip, sunflower, rose, dandelion, iris were selected as the classes
    global classNames
    for _class in classNames:
        count = 0
        currX = []
        currY = []
        for file_name in os.listdir(os.path.join(image_directory, _class)):
            count += 1
            img = Image.open(os.path.join(image_directory, _class, file_name))
            img = img.resize(IMAGE_SIZE)    
            if (img.mode != "RGB"):
                count -= 1
                continue
            #if (count % 50 == 0):
                #img.show()

            img_array = np.array(img).reshape(IMAGE_SIZE[0]*IMAGE_SIZE[1]*3,) / 255.0 - 0.5
            currX.append(img_array)
            currY.append(_class)

            print(f"Loaded {_class}/{file_name}")
            if (count == 986): # 986 images is minimum in rose folder
                break

        # split the data set equally
        currX_train, currX_test, currY_train, currY_test = train_test_split(currX, currY, test_size=0.2, random_state=42)
        for element in currX_train:
            X_train.append(element)
        
        for element in currX_test:
            X_test.append(element)
        
        for element in currY_train:
            Y_train.append(element)
        
        for element in currY_test:
            Y_test.append(element) 

    # one hot encoding
    encoder = OneHotEncoder()
    Y_test = encoder.fit_transform(np.array(Y_test).reshape(-1, 1)).toarray()
    Y_train = encoder.fit_transform(np.array(Y_train).reshape(-1, 1)).toarray()
    X_train = np.array(X_train)
    X_test = np.array(X_test)


    print("\n\nData loaded successfully\n\n")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print("\n\n")

    return X_train, X_test, Y_train, Y_test


# Load the data
X_train, X_test, Y_train, Y_test = load_train_data()
Y_test = Y_test.T
Y_train = Y_train.T
X_train = X_train.T
X_test = X_test.T

np.random.seed(19)


# Define the model
model = DLModel("Flower type Classifier")
model.add(DLLayer("Hidden Layer 1",  512, (X_train.shape[0],), activation="relu", W_initialization="He", learning_rate=0.2, random_scale=0.01))
model.add(DLLayer("Hidden Layer 2",  128, (512,), activation="trim_sigmoid", W_initialization="He", learning_rate=0.15, random_scale=0.01))
model.add(DLLayer("Hidden Layer 3",  64, (128,), activation="trim_tanh", W_initialization="He", learning_rate=0.1, random_scale=0.01))
model.add(DLLayer("Hidden Layer 3",  len(classNames), (64,), activation="trim_softmax", W_initialization="He", learning_rate=0.1, random_scale=0.01))
model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 1500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

# Calculate the accuracy
train_accuracy = np.sum(np.argmax(model.predict(X_train), axis=0) == np.argmax(Y_train, axis=0)) / Y_train.shape[1]
test_accuracy = np.sum(np.argmax(model.predict(X_test), axis=0) == np.argmax(Y_test, axis=0)) / Y_test.shape[1]

print("train accuracy:", train_accuracy)
print("test accuracy:", test_accuracy)

# example
model.confusion_matrix(X_test, Y_test)

model.save_weights(f"saved_weights {round(test_accuracy, 4) * 100}%")
