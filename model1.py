from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_train_data():
    df = pd.read_csv("dataset\\train.csv")
    

    df = df.dropna().drop_duplicates()

    # Normalize the data to range [-1, 1]
    scaler = StandardScaler()

    Y = df["price_range"].to_numpy()
    X = scaler.fit_transform(df.drop(columns=["price_range"]))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test


# Load the data
X_train, X_test, Y_train, Y_test = load_train_data()
X_train = X_train.T
X_test = X_test.T

np.random.seed(23)


# Define the model
model = DLModel("Price Range Classifier")
model.add(DLLayer("Hidden Layer 1",  256, (X_train.shape[0],), activation="relu", W_initialization="random", learning_rate=0.01, random_scale=0.01))
model.add(DLLayer("Hidden Layer 2", 512, (256,), activation="relu", W_initialization="random", learning_rate=0.01, random_scale=0.01))
model.add(DLLayer("Hidden Layer 4", 128, (512,), activation="trim_sigmoid", W_initialization="random", learning_rate=0.01, random_scale=0.01))
model.add(DLLayer("Output Layer", 4, (1280,), activation="trim_softmax", W_initialization="random", learning_rate=0.01, random_scale=0.01))
model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 1000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

print("train accuracy:", np.mean(model.predict(X_train) == Y_train))
print("test accuracy:", np.mean(model.predict(X_test) == Y_test))