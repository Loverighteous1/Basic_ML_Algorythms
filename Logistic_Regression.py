import numpy as np
import matplotlib.pyplot as plt





class LogisticRegression:
  '''
  The logistic regression is used when we want separate input data into 2 classes, commonly known as binary classification problems.
  With this class, the machine will learn with an output data called  binary target.
  '''
# Constructor
  
  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

# A column of one should be added to the features to enable the manipulation of the bias term 
    
  def add_ones(self, x):
    return np.hstack((np.ones((x.shape[0],1)),x))
    
# The logistic function expression
  def sigmoid(self, x):
    z = x @ self.w
    return np.divide( 1, 1 + np.exp(-z))

# Negative log likelihood function or cross entropy loss
  
  def cross_entropy(self, x, y_true):
    y_pred = self.sigmoid(x)
    loss = (1/x.shape[0]) * (-np.sum((y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))))
    return loss

#This function will use the sigmoid function to compute the probalities
  
  def predict_proba(self,x):  
    proba = self.sigmoid(x)
    return proba

  
  def predict(self,x):
    '''
     Predict the binary ooutput based on the input.
     Returns 1 if the probability is greater than or equal to 0.5 or 0 otherwise
    '''
    x = self.add_ones(x)
    probas = self.predict_proba(x)
    output = (probas >= .5).astype(int) #convert the probalities into 0 and 1 by using a treshold=0.5
    return output

# This fit method will be used to train the logistic regression on the given data 
  
  def fit(self,x,y):
    x = self.add_ones(x)                    

    #  Need of reshaping y
    y_true =  y.reshape((-1, 1))        

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w = np.zeros((x.shape[1],1))          

    for epoch in range(self.n_epochs):
      # make predictions
      y_pred = self.predict_proba(x)          

      #compute the gradient
      grad = - np.divide(x.T @ (y_true - y_pred), x.shape[0])     

      #update rule
      self.w = self.w - self.lr * grad              

      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y_true)            
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')

# To computes the accuracy 
        
  def accuracy(self,y_true, y_pred):
    '''It will evaluate the performance of the model
    A high accuracy is indicating that the model is predicting well, while
    a low accuracy is saying uncorrect prediction.
    '''
    y_true = y_true.reshape((-1, 1))
    acc = np.mean(y_true == y_pred) * 100
    return acc


  def plot_decision_boundary(X, w, b,y_train):

    # z = w1x1 + w2x2 + w0
    # one can think of the decision boundary as the line x2=mx1+c
    # Solving we find m and c
    x1 = [X[:,0].min(), X[:,0].max()]
    m = -w[1]/w[2]
    c = -b/w[2]
    x2 = m*x1 + c

    # Plotting
    fig = plt.figure(figsize=(10,8))
    plt.scatter(X[:, 0], X[:, 1],c=y_train)
    plt.scatter(X[:, 0], X[:, 1], c=y_train)
    plt.xlim([-2, 3])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x1, x2, 'y-') 