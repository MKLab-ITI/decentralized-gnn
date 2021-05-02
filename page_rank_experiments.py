from data import importer
from device import Device, DecentralizedVariable
import numpy as np
from nn import MLP
from random import random
import os
import pickle

def onehot(label, num_classes):
    return np.array([1. if label is not None and label == i else 0. for i in range(num_classes)])

# create data
G, features, labels, training, validation, test = importer.load("pubmed")
training, validation = validation, training
num_classes = len(set(labels.values()))
onehot_labels = {u: onehot(labels[u] if u in training else None, num_classes) for u in G}
for u, v in list(G.edges()):
    G.add_edge(v, u)


class PropagationDevice(Device):
    def __init__(self, node, ML_predictions, labels):
        super().__init__(node)
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


f = MLP(len(list(features.values())[0]), num_classes)
print(len(f.save()))
if not os.path.exists('mlp.pickle'):
    best_loss = float('inf')
    interest = 100
    for epoch in range(2000):
        for u in training:
            pred = f(features[u], is_training=True)
            f.backpropagate(onehot_labels[u])
        f.learner_end_batch()
        loss = -np.sum(np.log(f(features[u])+1.E-12)*onehot(labels[u], num_classes)+(1-onehot(labels[u], num_classes))*np.log(1+1.E-12-f(features[u])) for u in validation).sum()
        if loss < best_loss:
            best_vars = f.save()
            best_loss = loss
            interest = 100
            print("Epoch", epoch, loss,
                  "Validation Accuracy", sum(1. if np.argmax(f(features[u])) == labels[u] else 0 for u in validation) / len(validation),
                  "Test Accuracy", sum(1. if np.argmax(f(features[u])) == labels[u] else 0 for u in test) / len(test))
        else:
            interest -= 1
            if interest == 0:
                break
    with open('mlp.pickle', 'wb') as file:
        pickle.dump(best_vars, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('mlp.pickle', 'rb') as file:
    best_vars = pickle.load(file)
    f.load(best_vars)
    print("Test Accuracy", sum(1. if np.argmax(f(features[u])) == labels[u] else 0 for u in test)/len(test))


accuracies = list()
devices = {u: PropagationDevice(u, f(features[u]), onehot_labels[u] if u in training else onehot_labels[u]) for u in G}
for epoch in range(60):
    messages = list()
    for u, v in G.edges():
        if random() <= 0.1:
            message = devices[u].send(devices[v])
            messages.append(len(pickle.dumps(message)))
            message = devices[v].receive(devices[u], message)
            messages.append(len(pickle.dumps(message)))
            message = devices[u].ack(devices[v], message)
    accuracy = sum(1. if devices[u].predict() == labels[u] else 0 for u in test) / len(test)
    print("Epoch", epoch, "Accuracy", accuracy, "message size", sum(messages) / float(len(messages)))
    accuracies.append(accuracy)
print("pubmed_gnn =", accuracies, ";")
