from propagate import DecentralizedVariable, MergeVariable, Device, TopoMergeVariable, PPRVariable
import numpy as np

def onehot(label, num_classes):
    return np.array([1. if label is not None and label == i else 0. for i in range(num_classes)])

def mse(x1, x2):
    mx1 = x1.max()
    mx2 = x2.max()
    if mx1 == 0 or mx2 == 0:
        return 1
    return np.sum((x1/mx1-x2/mx2)**2)/x1.shape[0]


class GossipDevice(Device):
    def __init__(self, node, predictor, features, labels, gossip_merge=MergeVariable):
        super().__init__()
        self.node = node
        self.labels = labels
        self.features = features
        self.predictor = predictor
        self.ML_predictions = self.predictor(self.features)
        self.errors = self.append(PPRVariable(labels))
        self.predictions = self.append(PPRVariable(self.ML_predictions, "FDiff", balance=1))
        #self.scaler = self.append(PPRVariable(np.sum(np.abs(self.labels - self.ML_predictions)) if self.is_training() else 0, "FDiff", balance=1))
        self.train_with_neighbors = True
        if gossip_merge is not None:
            for model_var in self.predictor.variables:
                self.append(DecentralizedVariable(model_var, gossip_merge))
        self.update_predictor()

    def is_training(self):
        return self.labels.sum() != 0

    def train(self):
        if len(self.vars) > 2:
            if self.is_training():
                self.predictor(self.features, is_training=True)
                self.predictor.backpropagate(self.labels)
                self.predictor.learner_end_batch()
            self.ML_predictions = self.predictor(self.features)

    def update_predictor(self):
        self.train()
        self.errors.set(self.labels - self.ML_predictions if self.is_training() else self.labels) # i.e. "else zero"
        self.predictions.set(self.ML_predictions+self.errors.get())

    def predict(self, propagation=True):
        if not propagation:
            return np.argmax(self.ML_predictions)
        return np.argmax(self.predictions.get())

    def ack(self, device, message):
        super().ack(device, message)
        self.update_predictor()


class EstimationDevice(GossipDevice):
    def __init__(self, node, predictor, features, labels):
        self.synthetic = dict()
        super().__init__(node, predictor, features, labels, None)

    def train(self):
        if self.is_training():
            self.predictor(self.features, is_training=True)
            self.predictor.backpropagate(self.labels)
        else:
            self.synthetic[self] = (self.features, self.ML_predictions)
        for features, synthetic_predictions in self.synthetic.values():
            self.predictor(features, is_training=True)
            self.predictor.backpropagate(synthetic_predictions)
        self.predictor.learner_end_batch()
        self.ML_predictions = self.predictor(self.features)

    def send(self, device):
        return super().send(device), (self.features, self.labels if self.is_training() else self.ML_predictions)

    def ack(self, device, message):
        message, synthetic = message
        self.synthetic[device] = synthetic
        super().ack(device, message)