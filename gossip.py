from propagate import DecentralizedVariable, Device
import numpy as np


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

    def predict(self):
        return np.argmax(self.vars[1].value)

    def ack(self, device, message):
        if not self.is_training_node:
            self.vars[1].set(self.ML_predictions+10*self.vars[0].value)
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
            self.vars.append(DecentralizedVariable(model_var.value))
        self.update_predictor()

    def update_predictor(self):
        for decentralized_var, model_var in zip(self.vars[2:], self.f.variables):
            model_var.value = decentralized_var.value
        if self.is_training_node:
            for _ in range(40):
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
            self.vars[1].set(self.ML_predictions+10*self.vars[0].value)
        super().ack(device, message)
        self.update_predictor()


class EstimationDevice(Device):
    def __init__(self, node, f, features, labels):
        super().__init__()
        self.node = node
        self.is_training_node = labels.sum() != 0
        self.labels = labels
        self.f = f
        self.features = features
        self.vars.append(DecentralizedVariable(0, "FDiff", balance=1))
        self.vars.append(DecentralizedVariable(labels))
        self.synthetic = dict()
        self.update_predictor()

    def update_predictor(self):
        for decentralized_var, model_var in zip(self.vars[2:], self.f.variables):
            model_var.value = decentralized_var.value
        for _ in range(40):
            if self.is_training_node:
                self.f(self.features, is_training=True)
                self.f.backpropagate(self.labels)
            for features, synthetic_predictions in self.synthetic.values():
                self.f(features, is_training=True)
                self.f.backpropagate(synthetic_predictions)
            self.f.learner_end_batch()

        self.ML_predictions = self.f(self.features)
        errors = self.labels-self.ML_predictions if self.is_training_node else self.labels
        self.vars[0].set(errors)
        self.vars[0].update()

    def synthesize(self):
        return self.features, onehot(np.argmax(self.ML_predictions), self.ML_predictions.shape[0])

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
        super().ack(device, message)
        self.synthetic[device] = synthetic
        self.update_predictor()