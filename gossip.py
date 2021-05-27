from propagate import DecentralizedVariable, Device
import numpy as np
import random

def onehot(label, num_classes):
    return np.array([1. if label is not None and label == i else 0. for i in range(num_classes)])


class PageRankDevice(Device):
    def __init__(self, node, ML_predictions, labels):
        super().__init__()
        self.node = node
        self.is_training_node = labels.sum() != 0
        self.labels = labels
        self.ML_predictions = ML_predictions
        errors = labels-ML_predictions if self.is_training_node else labels
        self.vars.append(DecentralizedVariable(errors, "FDiff", balance=1))
        self.vars.append(DecentralizedVariable(labels))

    def predict(self, diffused=True):
        if not diffused:
            return np.argmax(np.random.uniform(size=self.labels.shape))
        return np.argmax(self.vars[1].value)

    def ack(self, device, message):
        if not self.is_training_node:
            self.vars[1].set(self.ML_predictions+self.vars[0].value)
        else:
            self.vars[1].set(self.labels)
        super().ack(device, message)


class GossipDevice(Device):
    def __init__(self, node, f, features, labels):
        super().__init__()
        self.node = node
        self.labels = labels
        self.f = f
        self.features = features
        self.vars.append(DecentralizedVariable(0, "FDiff", balance=1))
        self.vars.append(DecentralizedVariable(labels))
        self.is_training_node = labels.sum() != 0
        for model_var in self.f.variables:
            self.vars.append(DecentralizedVariable(model_var.value, lambda n, p: n if not self.is_training_node else 0.9*n+0.1*p))
        self.update_predictor()

    def update_predictor(self):
        for decentralized_var, model_var in zip(self.vars[2:], self.f.variables):
            model_var.value = decentralized_var.value
        if self.is_training_node:
            self.f(self.features, is_training=True)
            self.f.backpropagate(self.labels)
            self.f.learner_end_batch()
        for decentralized_var, model_var in zip(self.vars[2:], self.f.variables):
            decentralized_var.set(model_var.value)
            decentralized_var.update()
        self.ML_predictions = self.f(self.features)
        errors = self.labels - self.ML_predictions if self.is_training_node else self.labels
        self.vars[0].set(errors)
        self.vars[0].update()

    def predict(self, propagation=True):
        if not propagation:
            return np.argmax(self.ML_predictions)
        return np.argmax(self.vars[1].value)

    def ack(self, device, message):
        if not self.is_training_node:
            self.vars[1].set(self.ML_predictions+self.vars[0].value)
        super().ack(device, message)
        self.update_predictor()

def mse(x1, x2):
    x1 = x1-x1.min()
    mx = x1.max()
    if mx == 0:
        return 1
    x1 = x1/mx
    return np.sum((x1-x2)**2)/x1.shape[0]


class EstimationDevice(Device):
    def __init__(self, node, f, features, labels):
        super().__init__()
        self.node = node
        self.is_training_node = labels.sum() != 0
        self.labels = labels
        self.ML_predictions = labels
        self.f = f
        self.features = features
        self.patience = 200
        self.vars.append(DecentralizedVariable(0, "FDiff", balance=1))
        self.vars.append(DecentralizedVariable(labels))
        self.synthetic = dict()
        self.update_predictor()

    def update_predictor(self):
        for decentralized_var, model_var in zip(self.vars[2:], self.f.variables):
            model_var.value = decentralized_var.value
        #for _ in range(40):
        msqrt = 1 if len(self.synthetic)==0 else sum(mse(self.f(features), synthetic_predictions)**0.5 for features, synthetic_predictions in self.synthetic.values())
        msqrt = msqrt / max(len(self.synthetic),1)
        if msqrt > 0.1:
            if self.is_training_node:
                self.f(self.features, is_training=True)
                self.f.backpropagate(self.labels)
            #else:
            #    self.f(self.features, is_training=True)
            #    self.f.backpropagate(onehot(np.argmax(self.labels if self.is_training_node else self.ML_predictions), self.ML_predictions.shape[0]))
            for features, synthetic_predictions in self.synthetic.values():
                self.f(features, is_training=True)
                self.f.backpropagate(synthetic_predictions)
            self.f.learner_end_batch()
            #self.patience -= 1
            self.ML_predictions = self.f(self.features)
        errors = self.labels-self.ML_predictions if self.is_training_node else self.labels
        self.vars[0].set(errors)
        self.vars[0].update()

    def synthesize(self):
        return self.features, onehot(np.argmax(self.labels if self.is_training_node else self.ML_predictions), self.ML_predictions.shape[0])
        #else:
        #    device = random.choice(list(self.synthetic.keys())) # avoids creating tensor copies and makes calculations faster
        #    features = self.synthetic[device][0]
        #    return features, onehot(np.argmax(self.f(features)), self.ML_predictions.shape[0])

    def predict(self, propagation=True):
        if not propagation:
            return np.argmax(self.ML_predictions)
        return np.argmax(self.vars[1].value)

    def send(self, device):
        return super().send(device), self.synthesize()

    def ack(self, device, message):
        message, synthetic = message
        if not self.is_training_node:
            self.vars[1].set(self.ML_predictions+10*self.vars[0].value)
        #device = synthetic[0]
        #synthetic = synthetic[1], synthetic[2]
        self.synthetic[device] = synthetic
        self.update_predictor()
        super().ack(device, message)