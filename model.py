from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def load_train_data():
    df = pd.read_csv("dataset\heart_statlog_cleveland_hungary_final.csv")
    df = df.dropna()
    df = df.drop_duplicates()

    
    # Normalize the data
    scaler = MinMaxScaler()
    df[["fasting blood sugar", "resting bp s", "cholesterol", "max heart rate", "age"]] = scaler.fit_transform(df[["fasting blood sugar", "resting bp s", "cholesterol", "max heart rate", "age"]])

    Y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy()

    sns.heatmap(df.corr(), annot=True)
    plt.show()

    return X, Y

# Load the data
X, Y = load_train_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = X_train.T
X_test = X_test.T

np.random.seed(42)
print(len(np.unique(Y)))

# Define the model
model = DLModel("Obesity Classifier")
model.add(DLLayer("Hidden Layer 1", 248, (X_train.shape[0],), activation="leaky_relu", learning_rate=0.1))
model.add(DLLayer("Hidden Layer 2", 128, (248,), activation="tanh", learning_rate=0.01))
model.add(DLLayer("Output Layer", 1, (128,), activation="sigmoid", learning_rate=0.01))
model.compile("cross_entropy")
costs = model.train(X_train, Y_train, 10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()

print("train accuracy:", np.mean(model.predict(X_train) == Y_train))
print("test accuracy:", np.mean(model.predict(X_test) == Y_test))