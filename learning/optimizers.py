import numpy as np


class Variable:
    def __init__(self, value, regularization=0.005):
        self.value = value
        self.regularization = regularization


class Gradient:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, variable: Variable, error):
        variable.value = variable.value - error*self.learning_rate - variable.value*variable.regularization*self.learning_rate


class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1.E-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mt = dict()
        self.vt = dict()
        self.beta1t = dict()
        self.beta2t = dict()

    def update(self, variable: Variable, error):
        if variable not in self.mt:
            self.mt[variable] = 0
            self.vt[variable] = 0
            self.beta1t[variable] = 1
            self.beta2t[variable] = 1
        self.beta1t[variable] *= self.beta1
        self.beta2t[variable] *= self.beta2
        error = error + variable.value*variable.regularization
        self.mt[variable] = self.beta1*self.mt[variable] + (1-self.beta1)*error
        self.vt[variable] = self.beta2*self.vt[variable] + (1-self.beta2)*np.square(error)
        learning_ratet = self.learning_rate * (1-self.beta2t[variable])**0.5/(1-self.beta1t[variable])
        epsilont = self.epsilon * (1-self.beta1t[variable]) / (1-self.beta2t[variable])**0.5
        variable.value = variable.value - learning_ratet*self.mt[variable] / (epsilont+np.sqrt(self.vt[variable]))


class BatchOptimizer:
    def __init__(self, base):
        self.base = base
        self.accumulation = dict()
        self.sample_weight = 1

    def set_sample_weight(self, weight):
        self.sample_weight = weight

    def update(self, variable: Variable, error):
        self.accumulation[variable] = self.accumulation.get(variable, 0) + error * self.sample_weight

    def end_batch(self):
        for variable in self.accumulation:
            self.base.update(variable, self.accumulation[variable])
        self.accumulation = dict()