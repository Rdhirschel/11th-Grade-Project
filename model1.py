from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

IMAGE_SIZE = (128, 128)

def load_train_data(image_directory="dataset\\train"):
    X, Y = [], []

    classNames = os.listdir(image_directory)
    for folder_name in os.listdir(image_directory):
        count = 0
        for file_name in os.listdir(os.path.join(image_directory, folder_name)):
            count += 1
            img = Image.open(os.path.join(image_directory, folder_name, file_name))
            img = img.resize(IMAGE_SIZE)    
            if (img.mode != "RGB"):
                img = img.convert("RGB")


            img_array = np.array(img).reshape(IMAGE_SIZE[0]*IMAGE_SIZE[1]*3,) / 255.0 - 0.5
            X.append(img_array)
            Y.append(folder_name)

            print(f"Loaded {folder_name}/{file_name}")
            if (count == 723): # 723 images is minimum in astlibe folder, and we want to keep the data balanced
                break

    # one hot encoding
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    X = np.array(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=42)

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
model.add(DLLayer("Hidden Layer 1",  40, (X_train.shape[0],), activation="trim_sigmoid", W_initialization="random", learning_rate=0.1, random_scale=0.01))
model.add(DLLayer("Output Layer", 14, (40,), activation="trim_softmax", W_initialization="random", learning_rate=0.05, random_scale=0.01))
model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 1_000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

print("train accuracy:", np.mean(model.predict(X_train) == Y_train))
print("test accuracy:", np.mean(model.predict(X_test) == Y_test))

# example
print(model.predict(X_test[0, :]))
print(Y_test[0, :])
model.confusion_matrix(X_test, Y_test)

model.save_weights("saved_weights")
