import numpy as np


class Variable:
    datatype = np.float64

    def __init__(self, value, regularization=0.005):
        self.value = np.array(value, dtype=Variable.datatype)
        self.regularization = regularization


class Gradient:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, variable: Variable, error):
        variable.value = (
            variable.value
            - error * self.learning_rate
            - variable.value * variable.regularization * self.learning_rate
        )


class CenteredOptimizer:
    def __init__(self, base):
        self.base = base
        self.center = dict()

    def set_sample_weight(self, weight):
        self.base.set_sample_weight(weight)

    def update(self, variable: Variable, error):
        prev_center = self.center.get(variable, 0)
        error = error - 0.5 * (variable.value - prev_center)
        self.base.update(variable, error)
        self.center[variable] = variable.value

    def end_batch(self):
        self.base.end_batch()
        for variable in self.center:
            self.center[variable] = variable.value


class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1.0e-7):
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
        error = error + variable.value * variable.regularization
        self.mt[variable] = np.array(
            self.beta1 * self.mt[variable] + (1 - self.beta1) * error,
            dtype=Variable.datatype,
        )
        self.vt[variable] = np.array(
            self.beta2 * self.vt[variable] + (1 - self.beta2) * np.square(error),
            dtype=Variable.datatype,
        )
        learning_ratet = (
            self.learning_rate
            * (1 - self.beta2t[variable]) ** 0.5
            / (1 - self.beta1t[variable])
        )
        epsilont = (
            self.epsilon
            * (1 - self.beta1t[variable])
            / (1 - self.beta2t[variable]) ** 0.5
        )
        variable.value = np.array(
            variable.value
            - learning_ratet
            * self.mt[variable]
            / (epsilont + np.sqrt(self.vt[variable])),
            dtype=Variable.datatype,
        )


class BatchOptimizer:
    def __init__(self, base):
        self.base = base
        self.accumulation = dict()
        self.sample_weight = 1

    def set_sample_weight(self, weight):
        self.sample_weight = weight

    def update(self, variable: Variable, error):
        self.accumulation[variable] = (
            self.accumulation.get(variable, 0) + error * self.sample_weight
        )

    def end_batch(self):
        for variable in self.accumulation:
            self.base.update(variable, self.accumulation[variable])
        self.accumulation = dict()
