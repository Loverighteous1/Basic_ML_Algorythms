# HERE ALL THE FILES WILL BE CALLED OR IMPORTED
from Linear_Regression import LinearReg
from Logistic_Regression import LogisticReg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0) # To demonstrate that if we use the same seed value twice, we will get the same random number twice

  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test_split(X,y)



np.random.seed(10)
xtrain = np.linspace(0,1, 10)
ytrain = xtrain + np.random.normal(0, 0.1, (10,))

xtrain = xtrain.reshape(-1, 1)
ytrain = ytrain.reshape(-1, 1)


def main():
    inpu=input("Choose yes for Linear Regression and No for Logistic Regreesion :")
    if inpu=="yes":
        linear_reg = LinearReg(xtrain, ytrain, lr=float(input("Enter the learning rate")), epochs=int(input("Enter the number of Epoch")))
        linear_reg.fit()
    else:
        logistic_reg = LogisticReg(X_train, y_train, lr=float(input("Enter the learning rate")), epochs=int(input("Enter the number of Epoch")))
        logistic_reg.fit()

if __name__ =="__main__":
    main()

