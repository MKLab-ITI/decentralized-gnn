from data import importer
import gossip
from learning.nn import MLP
from random import random
from tqdm import tqdm
import pickle
from random import choice
import threading, time

dataset = "pubmed"
scheme = "synth"

# load data
G, features, labels, training, validation, test = importer.load(dataset)
training, validation = validation, training
#test = training
#training = list(set(range(len(labels)))-set(training))#
num_classes = len(set(labels.values()))
num_features = len(list(features.values())[0])
onehot_labels = {u: gossip.onehot(labels[u], num_classes) for u in G}
empty_label = gossip.onehot(None, num_classes)
for u, v in list(G.edges()):
    G.add_edge(v, u)

if "gossip" in scheme:
    devices = {u: gossip.GossipDevice(u, MLP(num_features, num_classes), features[u], onehot_labels[u] if u in training else empty_label) for u in G}
elif "synth" in scheme:
    devices = {u: gossip.EstimationDevice(u, MLP(num_features, num_classes), features[u], onehot_labels[u] if u in training else empty_label) for u in G}
elif "pretrained" in scheme:
    from predict import train_or_load_MLP
    f = train_or_load_MLP(dataset, features, onehot_labels, num_classes, training, validation, test)
    devices = {u: gossip.PageRankDevice(u, f(features[u]), onehot_labels[u] if u in training else empty_label) for u in G}
elif "pagerank" in scheme:
    devices = {u: gossip.PageRankDevice(u, onehot_labels[u] if u in training else empty_label, onehot_labels[u] if u in training else empty_label) for u in G}
else:
    raise Exception("Invalid scheme")


device_list = list(devices.values())
accuracies = list()
for epoch in range(100):
    threads = list()
    messages = list()
    messages.append(0)
    for u, v in tqdm(G.edges()):
        if random() <= 0.1 and u!=v:
            message = devices[u].send(devices[v])
            if "random" in scheme:  # but not ngossip
                message = message[0:2] + (choice(device_list).send(v)[2:])
            # messages.append(len(pickle.dumps(message)))
            message = devices[v].receive(devices[u], message)
            if "random" in scheme:  # but not ngossip
                message = message[0:2] + (choice(device_list).send(u)[2:])
            # messages.append(len(pickle.dumps(message)))
            devices[u].ack(devices[v], message)
    accuracy_base = sum(1. if devices[u].predict(False) == labels[u] else 0 for u in test) / len(test)
    accuracy = sum(1. if devices[u].predict() == labels[u] else 0 for u in test) / len(test)
    print("Epoch", epoch, "Accuracy", accuracy_base, '->', accuracy, "message size", sum(messages) / float(len(messages)))
    accuracies.append(accuracy)
print(dataset+"_"+scheme, accuracies, ";")
max_acc = max(accuracies)
best_epoch = None
for epoch in range(len(accuracies)):
    if best_epoch is None and abs(accuracies[epoch]-max_acc)<=0.001:
        best_epoch = epoch
    elif abs(accuracies[epoch]-max_acc) > 0.001:
        best_epoch is None
print("Epochs to converge", best_epoch+1)
