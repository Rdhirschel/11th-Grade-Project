import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns

def load_train_data():
    df= pd.read_csv('dataset\ObesityDataSet_raw_and_data_sinthetic.csv')

    df = df.dropna()
    df = df.drop_duplicates()

    # Minmax Scaler - normalizes the data to range [0,1] -> [-0.5, 0.5] after the -0.5
    mms = MinMaxScaler()
    df[["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "TUE", "FAF"]] = mms.fit_transform(df[["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "TUE", "FAF"]]) - 0.5

    lb = LabelEncoder()
    df[["Gender", "CALC", "FAVC", "SCC", "SMOKE", "family_history_with_overweight", "MTRANS", "CAEC", "NObeyesdad"]] = df[["Gender", "CALC", "FAVC", "SCC", "SMOKE", "family_history_with_overweight", "MTRANS", "CAEC", "NObeyesdad"]].apply(lb.fit_transform)

    Y = df["NObeyesdad"].to_numpy()
    X = df.drop(columns=["NObeyesdad"]).to_numpy()
    return X, Y

# Load the data
X, Y = load_train_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
X_train = X_train.T
X_test = X_test.T

np.random.seed(1)

# Define the model
model = DLModel("Obesity Classifier")
model.add(DLLayer("Hidden Layer 1", 64, (X_train.shape[0],), "relu", "random", 0.001, random_scale=0.1))
model.add(DLLayer("Hidden Layer 2", 32, (64,), "sigmoid", "random", 0.006, random_scale=0.1))
model.add(DLLayer("Output Layer", len(np.unique(Y)), (32,), "trim_softmax", "He", 0.025))

model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

print("train accuracy:", np.mean(model.predict(X_train) == Y_train))
print("test accuracy:", np.mean(model.predict(X_test) == Y_test))