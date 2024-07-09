import numpy as np

class Adam(object):
    def __init__(self, n_iter=100, lr=0.1, beta1=0.8, beta2=0.999, eps=1e-8):
        self.n_iter = n_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def initialize_adam(self, parameters):
        L = len(parameters)
        v = np.zeros(L)
        s = np.zeros(L)
        self.v = v
        self.s = s

    def update_parameters_with_adam(self, parameters, grads, t):
        self.s = self.beta1 * self.s + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grads ** 2
        s_hat = self.s / (1.0 - self.beta1 ** (t + 1))
        v_hat = self.v / (1.0 - self.beta2 ** (t + 1))

        parameters = parameters - self.lr * s_hat / (np.sqrt(v_hat) + self.eps)
        return parameters
        
