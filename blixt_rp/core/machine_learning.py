import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    A common activation function
    """
    return 1. / (1. + np.exp(-z))

class machine_learning(object):
    """
    Based on 
    https://github.com/dennisbakhuis/Tutorials/blob/master/Logistic_Regression/Logistic_Regression.ipynb
    Conventions
    x: the feature vector
    w: weights vector
    b: bias
    act_func: activation function

    """

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x, w, b, act_func=sigmoid):
        """
        Predicted outcome of x, for given w & b
        """
        return act_func(np.dot(w.T, x) + b)

    def loss(self, a, y, metric='bce', epsilon=1e-15):
        """
        Calculates the error using the binary cross entropy function
        The added error 'epsilon' is added to avoid the computation of log(0)
        """
        e = epsilon
        if metric == 'bce':
            return (-1./len(a)) * np.sum(
                y * np.log(a + e) + (1. - y) * np.log(1. - a + e)
                )
        else:
            raise NotImplementedError("Only binary cross entropy error")

    def loss_gradient(self, x, y, a, metric='bce'):
        m = len(y)
        if metric == 'bce':
            dw = 1./m * np.dot(x, (a - y).T)
            db = 1./m * np.sum(a - y)
            return dw, db
        else:
            raise NotImplementedError("Only binary cross entropy error")

    def update(self, w, b, dw, db, learning_rate=0.01):
        w = w - learning_rate * dw 
        b = b - learning_rate * db 
        return w, b

    def round_limit(self, a, limit=0.5):
        """
        The activation function returns a probability between 0 and 1
        By definition, values <= 0.5 are rounded to 0 and values > 0.5 are rounded to 1
        """
        return np.uint8( a > limit)

    def accuracy(self, y_hat, y):
        return round(np.sum(y_hat==y) / len(y_hat) * 1000)/10.

if __name__ == '__main__':
    import pandas as pd

    num_iterations = 1000
    lr = 0.01
    df = pd.read_csv('C:\\Users\\marten\\Downloads\\train_data.csv')
    df = df.drop(['Unnamed: 0', 'PassengerId'], axis=1)
    x = df.iloc[:,1:].to_numpy() # features (rows = observations, columns = features) 
    x = x.T # Need to transpose it
    y = df['Survived'].to_numpy() # outcome

    np.random.seed(2020)
    w = 0.01 * np.random.randn(14)
    b = 0.

    losses, acces = [], []
    ml = machine_learning()

    for i in range(num_iterations):
        a = ml.forward(x, w, b)
        err = ml.loss(y, a)
        y_hat = ml.round_limit(a)
        acc = ml.accuracy(y_hat, y)
        dw, db = ml.loss_gradient(x, y, a)
        w, b = ml.update(w, b, dw, db, learning_rate=lr)
        losses.append(err)
        acces.append(acc)
        if i % 100 == 0:
            print("Error:", err, f'\tAccuracy:', acc)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(losses)), losses, 'b-', label='loss')
    xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('loss') 

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(acces)), acces, 'b-', label='accuracy')
    _, _ = ax.set_xlabel('epoch'), ax.set_ylabel('accuracy')
    print("Weights:", w)

    plt.show()
