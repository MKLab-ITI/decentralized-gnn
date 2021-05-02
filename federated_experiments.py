from data import importer
from device import Device, DecentralizedVariable
import numpy as np
from nn import MLP
from random import random, choice
from tqdm import tqdm
import os
import pickle

def onehot(label, num_classes):
    return np.array([1. if label is not None and label == i else 0. for i in range(num_classes)])

# create data
G, features, labels, training, validation, test = importer.load("pubmed")
training, validation = validation, training
num_classes = len(set(labels.values()))
num_features = len(list(features.values())[0])
onehot_labels = {u: onehot(labels[u] if u in training else None, num_classes) for u in G}
for u, v in list(G.edges()):
    G.add_edge(v, u)


class PropagationDevice(Device):
    def __init__(self, node, f, features, labels):
        super().__init__(node)
        self.is_training_node = labels.sum() != 0
        self.labels = labels
        self.f = f
        self.features = features
        self.vars.append(DecentralizedVariable(0, "FDiff", balance=1))
        self.vars.append(DecentralizedVariable(labels))
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
        errors = self.labels-self.ML_predictions if self.is_training_node else self.labels
        self.vars[0].set(errors)
        self.vars[0].update()

    def predict(self):
        return np.argmax(self.ML_predictions)
        #return np.argmax(self.vars[1].value)

    def ack(self, device, message):
        if not self.is_training_node:
            self.vars[1].set(self.ML_predictions+10*self.vars[0].value)
        super().ack(device, message)
        self.update_predictor()


devices = {u: PropagationDevice(u, MLP(num_features, num_classes), features[u], onehot_labels[u] if u in training else onehot_labels[u]) for u in G}
device_list = list(devices.values())
accuracies = list()
for epoch in range(100):
    messages = list()
    for u, v in tqdm(G.edges()):
        if random() <= 0.1:
            message = devices[u].send(devices[v])
            message = message[0:2] + (choice(device_list).send()[2:])
            messages.append(len(pickle.dumps(message)))
            message = devices[v].receive(devices[u], message)
            message = message[0:2] + (choice(device_list).send()[2:])
            messages.append(len(pickle.dumps(message)))
            message = devices[u].ack(devices[v], message)
    accuracy = sum(1. if devices[u].predict() == labels[u] else 0 for u in test)/len(test)
    print("Epoch", epoch, "Accuracy", accuracy, "message size", sum(messages)/float(len(messages)))
    accuracies.append(accuracy)
print("pubmed_gossip =", accuracies, ";")