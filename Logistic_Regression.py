import numpy as np
import matplotlib.pyplot as plt


class LogisticReg:
    def __init__(self, x, y, lr, epochs):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.param_set = None
        self.losses = []

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

    def add_ones(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))
    
    def sigmoid(self, x):
        if x.shape[1] != self.param_set.shape[0]:
            x = self.add_ones(x)
        z = x @ self.param_set
        return 1/(1+np.exp(-z))
    
    def nll_criteria(self, y, y_pred): # aka cross entropy 
        nll = -(1/y.shape[0]) * np.sum(y * np.log(y_pred) + (1 - y) *  np.log(1 - y_pred))
        return nll
        
    def grad_loss(self, x, y):
        return (-1/x.shape[0]) * x.T @ (y - self.sigmoid(x))

    def batch_grad_desc(self):
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        for iteration in range(self.epochs):
            y_pred = self.sigmoid(x)
            loss = self.nll_criteria(self.y, y_pred)
            grad = self.grad_loss(x, self.y)
            # Upadate rule to find the "best" set of param (or weights)
            self.param_set = self.param_set - self.lr * grad
            self.losses.append(loss)
            #print(self.losses)
        plt.plot(self.losses)
        plt.show()
        #return self.param_set

    def batch_grad_desc_mom(self, beta):
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        self.momentum = 0
        for iteration in range(self.epochs):
            y_pred = self.sigmoid(x)
            loss = self.nll_criteria(self.y, y_pred)
            grad = self.grad_loss(x, self.y)
            self.momentum = beta * self.momentum + (1 - beta) * grad
            # Upadate rule to find the "best" set of param (or weights)
            self.param_set = self.param_set - self.lr * self.momentum
            self.losses.append(loss)

    def minibatch_GD(self, batch_size):
        self.batch = batch_size
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        for iteration in range(self.epochs):
            x , self.y = self.shuffle_data(x , self.y)
            running_losses = 0
            for batch in range(0, self.x.shape[0], self.batch):
                batch_x = x[batch: batch + self.batch]
                batch_y = self.y[batch: batch + self.batch]
                y_pred = self.sigmoid(batch_x)
                loss = self.nll_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * grad
                running_losses += loss
            self.losses.append(running_losses/self.x.shape[0])

    def minibatch_GD_mom(self, batch_size, beta):
        self.batch = batch_size
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        for iteration in range(self.epochs):
            x , self.y = self.shuffle_data(x , self.y)
            running_losses = 0
            self.momentum = 0
            for batch in range(0, self.x.shape[0], self.batch):
                batch_x = x[batch: batch + self.batch]
                batch_y = self.y[batch: batch + self.batch]
                y_pred = self.sigmoid(batch_x)
                loss = self.nll_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                self.momentum = beta * self.momentum + (1 - beta) * grad
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * self.momentum
                running_losses += loss
            self.losses.append(running_losses/x.shape[0])

    
    def stochastic_GD(self):
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        avg_loss = float('inf') # inf is infinity
        tolerance = 0.0001
        epoch = 0
        while epoch < self.epochs and avg_loss:
            x , self.y = self.shuffle_data(x , self.y)
            running_losses = 0
            for idx in range(0, x.shape[0]):
                batch_x = x[idx].reshape(-1, x.shape[1])
                batch_y = self.y[idx].reshape(-1, 1)
                y_pred = self.sigmoid(batch_x)
                loss = self.nll_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * grad
                running_losses += loss
            avg_loss = running_losses/self.x.shape[0]
            self.losses.append(avg_loss)
            epoch += 1

    def stochastic_GD_mom(self, beta):
        x = self.add_ones(self.x)
        self.param_set = np.zeros((x.shape[1], 1))
        self.losses = []
        avg_loss = float('inf')
        tolerance = 0.0001
        epoch = 0
        while epoch < self.epochs and avg_loss:
            self.momentum = 0
            x , self.y = self.shuffle_data(x , self.y)
            running_losses = 0
            for idx in range(0, x.shape[0]):
                batch_x = x[idx].reshape(-1, x.shape[1])
                batch_y = self.y[idx].reshape(-1, 1)
                y_pred = self.sigmoid(batch_x)
                loss = self.nll_criteria(batch_y, y_pred)
                grad = self.grad_loss(batch_x, batch_y)
                self.momentum = beta * self.momentum + (1 - beta) * grad
                # Upadate rule to find the "best" set of param (or weights)
                self.param_set = self.param_set - self.lr * self.momentum
                running_losses += loss
            avg_loss = running_losses/x.shape[0]
            self.losses.append(avg_loss)
            epoch += 1


    def fit(self, batch=None, momentum=0):
        if batch is None:
            if momentum == 0:
                gd = self.batch_grad_desc()
                #self.param_set = gd.param_set
            else:
                gd = self.batch_grad_desc_mom(momentum)
                self.param_set = gd.param_set
        elif batch > 1:
            if momentum == 0:
                gd = self.minibatch_GD(batch)
                self.param_set = gd.param_set
            else:
                gd = self.minibatch_GD_mom(batch, momentum)
                self.param_set = gd.param_set
        else:
            if momentum == 0:
                gd = self.stochastic_GD()
                self.param_set = gd.param_set
            else:
                gd = self.stochastic_GD_mom(momentum)
                self.param_set = gd.param_set