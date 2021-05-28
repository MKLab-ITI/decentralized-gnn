import numpy as np
import inspect
from .optimizers import Variable, Adam, BatchOptimizer


class Derivable(object):
    def __init__(self, learner):
        self.learner = learner

    def __call__(self, inputs, is_training=False):
        self.inputs = inputs
        self.value = self._forward(inputs, is_training)
        return self.value

    def get_vars(self):
        return list()

    def backpropagate(self, error):
        return self._backward(self.inputs, self.value, error)

    def _forward(self, inputs, is_training=False):
        raise Exception("Not implemented")

    def _backward(self, inputs, value, error):
        raise Exception("Not implemented")

    def serialize(self):
        raise Exception("Not implemented")


class Affine(Derivable):
    def __init__(self, num_inputs, num_outputs, learner, bias=False, regularization=0.0005):
        super().__init__(learner)
        self.W = Variable(xavier(num_outputs, num_inputs), regularization=regularization)
        self.b = Variable(np.zeros(num_outputs), regularization=regularization) if bias else False

    def get_vars(self):
        return [self.W, self.b] if self.b else [self.W]

    def _forward(self, inputs, is_training=False):
        outputs = np.matmul(self.W.value, inputs)
        if self.b:
            outputs += self.b.value
        return outputs

    def _backward(self, inputs, value, error):
        derivative = np.matmul(self.W.value.transpose(), error)
        self.learner.update(self.W, np.outer(error, inputs))
        if self.b:
            self.learner.update(self.b, error)
        return derivative


class Relu(Derivable):
    def _forward(self, inputs, is_training=False):
        return inputs*(inputs>0)

    def _backward(self, inputs, value, error):
        return error*(inputs>0)


class SoftmaxCE(Derivable):
    def _forward(self, inputs, is_training=False):
        exps = np.exp(inputs - inputs.max())
        return exps / np.sum(exps)

    def _backward(self, inputs, value, desired_output):
        SM = value.reshape((-1, 1))
        softmax_jacobian = np.diagflat(value) - np.dot(SM, SM.T)
        derivative = np.matmul(-desired_output / (1.E-8 + value) + (1 - desired_output) / (1 - value + 1.E-8), softmax_jacobian)
        return derivative

class Dropout(Derivable):
    def __init__(self, dropout=0.5):
        self.dropout = dropout

    def _forward(self, inputs, is_training=False):
        self.mask = np.random.binomial(1, self.dropout, size=inputs.shape)/self.dropout if is_training else 1
        return inputs*self.mask

    def _backward(self, inputs, value, error):
        return error*self.mask


def xavier(d0, d1, he=2):
    return (np.random.rand(d0, d1)*2-1)*(6./(d0+d1))**0.5 / he


class MLP(Derivable):
    def __init__(self, num_inputs, num_outputs, learner=None):
        if learner is None:
            learner = BatchOptimizer(Adam())
        super().__init__(learner)
        hidden_units = 64
        self.variables = list()
        self.layers = list()
        self.append(Dropout(0.5))
        self.append(Affine(num_inputs, hidden_units, learner))
        self.append(Relu(learner))
        self.append(Dropout(0.5))
        self.append(Affine(hidden_units, num_outputs, learner, regularization=0))
        self.append(SoftmaxCE(learner))

    def append(self, layer):
        self.layers.append(layer)
        self.variables.extend(layer.get_vars())

    def __call__(self, features, is_training=False):
        features = np.array(features)
        for layer in self.layers:
            features = layer(features, is_training)
        return features

    def backpropagate(self, derivative):
        if np.sum(derivative) == 0:
            return
        for layer in self.layers[::-1]:
            derivative = layer.backpropagate(derivative)

    def save(self):
        return [var.value for var in self.variables]

    def load(self, values):
        for value, var in zip(values, self.variables):
            var.value = value

    def learner_end_batch(self):
        self.learner.end_batch()
