import numpy as np
import matplotlib.pyplot as plt
class LinearReg:
    def __init__(self, x, y, lr, epochs):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.param_set = None
        #self.losses = []

    def shuffle_data(self, X, y):
        """Shuffles the data
        Args:
        X: input features of size - N x D
        y: target vector of size - N x 1
        
        Returns:
        shuffled data
        """
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        return X[shuffled_idx], y[shuffled_idx]

    def linpred(self, x):
        return x @ self.param_set
    
    def mse_criteria(self, y, y_pred):
        mse = (1/y.shape[0]) * np.sum((y - y_pred)**2)
        return mse
        
    def grad_loss(self, x, y):
        return (-2/x.shape[0]) * x.T @ (y - self.linpred(x))
    
    def plot_loss(self, losses):
        plt.plot(losses)
        plt.show()

    def batch_grad_desc(self):
        self.param_set = np.zeros((self.x.shape[1], 1))
        losses = []
        for iteration in range(self.epochs):
            y_pred = self.linpred(self.x)
            loss = self.mse_criteria(self.y, y_pred)
            grad = self.grad_loss(self.x, self.y)
            # Upadate rule to find the "best" set of param (or weights)
            self.param_set = self.param_set - self.lr * grad
            losses.append(loss)
            print(f"Epoch {iteration} loss :{loss}")
            
        return self.plot_loss(losses)

    def batch_grad_desc_mom(self, beta):
        self.param_set = np.zeros((self.x.shape[1], 1))
        self.losses = []
        self.momentum = 0
        for iteration in range(self.epochs):
            y_pred = self.linpred(self.x)
            loss = self.mse_criteria(self.y, y_pred)
            grad = self.grad_loss(self.x, self.y)
            self.momentum = beta * self.momentum + (1 - beta) * grad
            # Upadate rule to find the "best" set of param (or weights)
            self.param_set = self.param_set - self.lr * self.momentum
            self.losses.append(loss)

    def minibatch_GD(self, batch_size):
        self.batch = batch_size
        self.param_set = np.zeros((self.x.shape[1], 1))
        self.losses = []
        for iteration in range(self.epochs):
            self.x , self.y = self.shuffle_data(self.x , self.y)
            running_losses = 0
            for batch in range(0, self.x.shape[0], self.batch):
                batch_x = self.x[batch: batch + self.batch]
                batch_y = self.y[batch: batch + self.batch]
                y_pred = self.linpred(batch_x)
                loss = self.mse_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * grad
                running_losses += loss
            self.losses.append(running_losses/self.x.shape[0])

    def minibatch_GD_mom(self, batch_size, beta):
        self.batch = batch_size
        self.param_set = np.zeros((self.x.shape[1], 1))
        self.losses = []
        for iteration in range(self.epochs):
            self.x , self.y = self.shuffle_data(self.x , self.y)
            running_losses = 0
            self.momentum = 0
            for batch in range(0, self.x.shape[0], self.batch):
                batch_x = self.x[batch: batch + self.batch]
                batch_y = self.y[batch: batch + self.batch]
                y_pred = self.linpred(batch_x)
                loss = self.mse_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                self.momentum = beta * self.momentum + (1 - beta) * grad
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * self.momentum
                running_losses += loss
            self.losses.append(running_losses/self.x.shape[0])

    
    def stochastic_GD(self):
        self.param_set = np.zeros((self.x.shape[1], 1))
        self.losses = []
        avg_loss = float('inf')
        tolerance = 0.0001
        epoch = 0
        while epoch < self.epochs and avg:
            self.x , self.y = self.shuffle_data(self.x , self.y)
            running_losses = 0
            for idx in range(0, self.x.shape[0]):
                batch_x = self.x[idx].reshape(-1, self.x.shape[1])
                batch_y = self.y[idx].reshape(-1, 1)
                y_pred = self.linpred(batch_x)
                loss = self.mse_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * grad
                running_losses += loss
            avg_loss = running_losses/self.x.shape[0]
            self.losses.append(avg_loss)
            epoch += 1

    def stochastic_GD_mom(self, beta):
        self.param_set = np.zeros((self.x.shape[1], 1))
        self.losses = []
        avg_loss = float('inf')
        tolerance = 0.0001
        epoch = 0
        while epoch < self.epochs and avg_loss:
            self.momentum = 0
            self.x , self.y = self.shuffle_data(self.x , self.y)
            running_losses = 0
            for idx in range(0, self.x.shape[0]):
                batch_x = self.x[idx].reshape(-1, self.x.shape[1])
                batch_y = self.y[idx].reshape(-1, 1)
                y_pred = self.linpred(batch_x)
                loss = self.mse_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                self.momentum = beta * self.momentum + (1 - beta) * grad
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * self.momentum
                running_losses += loss
            avg_loss = running_losses/self.x.shape[0]
            self.losses.append(avg_loss)
            epoch += 1
            
    def fit(self, batch=None, momentum=0):
        if batch is None:
            if momentum == 0:
                self.batch_grad_desc()
            else:
                self.batch_grad_desc_mom(momentum)
        elif batch > 1:
            if momentum == 0:
                self.minibatch_GD(batch)
            else:
                self.minibatch_GD_mom(batch, momentum)
        else:
            if momentum == 0:
                self.stochastic_GD()
            else:
                self.stochastic_GD_mom(momentum)