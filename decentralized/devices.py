import numpy as np
from learning.optimizers import Variable
from decentralized.abstracts import Device, DecentralizedVariable
from decentralized.mergers import AvgMerge, PPRVariable, Smoothen, DecoupleNormalization
import random


def mse(x1, x2):
    mx1 = x1.max()
    mx2 = x2.max()
    if mx1 == 0 or mx2 == 0:
        return 1
    return np.sum((x1/mx1-x2/mx2)**2)/x1.shape[0]


class GossipDevice(Device):
    def __init__(self, node, predictor, features, labels, gossip_merge=AvgMerge, train_steps=1):
        super().__init__()
        self.node = node
        self.labels = labels
        self.features = features
        self.predictor = predictor
        smoother = Smoothen#lambda x: x
        self._is_training = self.labels.sum() != 0
        self.ML_predictions = self.predictor(self.features) * (1 if gossip_merge is not None else 0)
        self.errors = self.append(smoother(PPRVariable(labels, "FDiff", balance=1, is_training=self._is_training)))
        self.predictions = self.append(smoother(PPRVariable(self.ML_predictions, "PPR", is_training=self._is_training)))
        #self.scaler = self.append(PPRVariable(np.sum(np.abs(self.labels - self.ML_predictions)) if self.is_training() else 0, "FDiff", balance=1))
        if gossip_merge is not None:
            for model_var in self.predictor.variables:
                self.append((DecentralizedVariable(model_var, lambda *args, **kwargs: (gossip_merge(*args, **kwargs)), is_training=self.is_training())))
        self.ML_predictions = self.predictor(self.features)
        self.model_updates = self.append(PPRVariable(1 if self._is_training else 0, "FDiff"))
        self.prediction_model = {"model": self.predictor, "value": self.model_updates.get()}
        self.update_predictor()
        self.train_steps = train_steps

    def is_training(self):
        return self._is_training

    def train(self):
        if len(self.vars) > 3:
            for _ in range(self.train_steps):
                if self.is_training():
                    self.predictor(self.features, is_training=True)
                    self.predictor.backpropagate(self.labels)
                    self.predictor.learner_end_batch()
            self.ML_predictions = self.prediction_model["model"](self.features)

    def update_predictor(self):
        self.errors.set(self.labels - self.ML_predictions if self.is_training() else self.labels) # i.e. "else zero"
        self.predictions.set(self.ML_predictions+self.errors.get())

    def predict(self, propagation=True):
        if not propagation:
            return np.argmax(self.ML_predictions)
        return np.argmax(self.predictions.get())

    def send(self, device=None):
        return super().send(device), self.prediction_model

    def ack(self, device, message):
        message, _ = message
        """if prediction_model["value"] > self.prediction_model["value"]:
            self.prediction_model = prediction_model
        if self.prediction_model["value"] < self.model_updates.get():
            self.prediction_model = {"model": self.predictor, "value": self.model_updates.get()}
        self.prediction_model = {"model": self.prediction_model["model"], "value": self.prediction_model["value"]}
        """
        super().ack(device, message)
        self.train()
        #super().ack(device, message)
        self.update_predictor()


class EstimationDevice(GossipDevice):
    def __init__(self, node, predictor, features, labels, gossip_merge=None):
        self.synthetic = dict()
        super().__init__(node, predictor, features, labels, None)
        from .mergers import RandomMergeVariable
        self.random_weight_var = DecentralizedVariable(Variable(1.), RandomMergeVariable, is_training=self.is_training())
        self.append(self.random_weight_var)

    def train(self):
        if self.is_training():
            self.synthetic[self] = (self.features, self.labels if self.is_training() else self.ML_predictions)
            for device in self.synthetic:
                if device in self.random_weight_var.merger.neighbor_weights or device==self:
                    features, synthetic_predictions = self.synthetic[device]
                    #self.predictor.learner.set_sample_weight(1. if device == self else self.random_weight_var.merger.neighbor_weights[device])
                    self.predictor(features, is_training=True)
                    self.predictor.backpropagate(synthetic_predictions)
            self.predictor.learner_end_batch()
        self.ML_predictions = self.predictor(self.features)
        #self.synthetic.clear()

    def send(self, device):
        return super().send(device), (self.features, self.ML_predictions)

    def ack(self, device, message):
        message, synthetic = message
        self.synthetic[device] = synthetic
        super().ack(device, message)


class CorpusDevice(GossipDevice):
    def __init__(self, node, predictor, features, labels, gossip_merge=None):
        self.synthetic = dict()
        self.prev_synthetic_len = 0
        super().__init__(node, predictor, features, labels, gossip_merge)

    def train(self):
        if self.is_training():
            self.synthetic[self] = (self.features, self.labels)
            if len(self.synthetic) > self.prev_synthetic_len*2:
                #print(self.prev_synthetic_len)
                self.prev_synthetic_len = len(self.synthetic)
                for _ in range(300):
                    for device in self.synthetic:
                        features, synthetic_predictions = self.synthetic[device]
                        self.predictor(features, is_training=True)
                        self.predictor.backpropagate(synthetic_predictions)
                    self.predictor.learner_end_batch()
        self.ML_predictions = self.predictor(self.features)

    def send(self, device):
        samples = random.sample(list(self.synthetic.keys()), min(len(self.synthetic), 5)) if self.synthetic else None
        return super().send(device), None if samples is None else [(sample, self.synthetic[sample][0], self.synthetic[sample][1]) for sample in samples]

    def ack(self, device, message):
        message, synthetics = message
        if synthetics is not None:
            for synthetic in synthetics:
                self.synthetic[synthetic[0]] = (synthetic[1], synthetic[2])
        super().ack(device, message)
