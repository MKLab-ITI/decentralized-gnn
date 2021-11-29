import numpy as np
from learning.optimizers import Variable
from decentralized.abstracts import Device, DecentralizedVariable
from decentralized.mergers import AvgMerge, PPRVariable, Smooth
import random


def mse(x1, x2):
    mx1 = x1.max()
    mx2 = x2.max()
    if mx1 == 0 or mx2 == 0:
        return 1
    return np.sum((x1/mx1-x2/mx2)**2)/x1.shape[0]


class GossipDevice(Device):
    def __init__(self, node, predictor, features, labels, gossip_merge=AvgMerge, smoothen=lambda x: x):
        super().__init__()
        self.node = node
        self.labels = labels
        self.features = features
        self.predictor = predictor
        self.ML_predictions = self.predictor(self.features) * (1 if gossip_merge is not None else 0)
        self.errors = self.append((PPRVariable(labels, "PPR")))
        self.predictions = self.append((PPRVariable(self.ML_predictions, "FDiff", balance=1)))
        self._is_training = self.labels.sum() != 0
        self.scaler = self.append(PPRVariable(np.sum(np.abs(self.labels - self.ML_predictions)) if self._is_training else 0, "FDiff", balance=1))
        if gossip_merge is not None:
            for model_var in self.predictor.variables:
                self.append(DecentralizedVariable(model_var, lambda *args, **kwargs: smoothen(gossip_merge(*args, **kwargs)), is_training=self._is_training))
        self.ML_predictions = self.predictor(self.features)
        self._update_predictor()

    def _train(self):
        if len(self.vars) > 3:
            if self._is_training:
                self.predictor(self.features, is_training=True)
                self.predictor.backpropagate(self.labels)
                self.predictor.learner_end_batch()
            self.ML_predictions = self.predictor(self.features)

    def _update_predictor(self):
        self.errors.set(self.labels - self.ML_predictions if self._is_training else self.labels) # i.e. "else zero"
        #norm = np.sum(np.abs(self.errors.get()))
        #if norm==0:
        #    norm = 1
        self.predictions.set(self.ML_predictions+self.errors.get())
        #if self._is_training:
        #    self.scaler.set(np.sum(np.abs(self.labels - self.ML_predictions)))

    def predict(self, propagation=True):
        if not propagation:
            return np.argmax(self.ML_predictions)
        return np.argmax(self.predictions.get())

    def ack(self, device, message):
        super().ack(device, message)
        self._train()
        #super().ack(device, message)
        self._update_predictor()

