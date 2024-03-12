import numpy as np
import matplotlib.pyplot as plt



class Linear_regression:
  def __init__(self, X, Yactual) -> None:
    self.X = X
    self.Yactual = Yactual

# Definition of the hypothesis
  def linreg_model(self, X, param_set, Ypred):
    self.Ypred = param_set.T @ self.X

# This MSE foncyion to evaluate the proximity of the regression line to the dataset points
  def mse_criteria(self,Ypred, Yactual, num_obs):
    mse_val = np.sum((self.Ypred - self.Yactual) ** 2) / num_obs
    return mse_val

# Obtain the gradient in order to find the local minimum which is the best set of parameters
  def d_lossfunc(self, Ypred, Yactual,  num_obs):
    grad_loss = np.sum((self.Ypred - self.Yactual)) / num_obs
    return grad_loss

# Solve the problem in both case: well-conditioned and ill-conditioned, use of GD algo
  def param_estimation(self, X, Yactual, param_size,learn_rate, epoch_size ):
    if np.linalg.det(self.X.T @ self.X) != 0:
      #Then the covariance matrix is invertible. The problem is well-conditioned. Hence easy to solve.
      param_set_mle = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
      return param_set_mle
    

  def prediction(self):
    pass