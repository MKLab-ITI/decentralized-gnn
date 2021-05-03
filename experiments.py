from data import importer
import gossip
from learning.nn import MLP
from random import random
from tqdm import tqdm
import pickle
from random import choice

dataset = "pubmed"
scheme = "gossip"

# load data
G, features, labels, training, validation, test = importer.load(dataset)
training, validation = validation, training
num_classes = len(set(labels.values()))
num_features = len(list(features.values())[0])
onehot_labels = {u: gossip.onehot(labels[u] if u in training else None, num_classes) for u in G}
for u, v in list(G.edges()):
    G.add_edge(v, u)

if "gossip" in scheme:
    devices = {u: gossip.GossipDevice(u, MLP(num_features, num_classes), features[u], onehot_labels[u] if u in training else onehot_labels[u]) for u in G}
elif "synth" in scheme:
    devices = {u: gossip.EstimationDevice(u, MLP(num_features, num_classes), features[u], onehot_labels[u] if u in training else onehot_labels[u]) for u in G}
elif "pretrained" in scheme:
    from predict import train_or_load_MLP
    f = train_or_load_MLP(dataset, features, onehot_labels, num_classes, training, validation, test)
    devices = {u: gossip.PageRankDevice(u, f(features[u]), onehot_labels[u] if u in training else onehot_labels[u]) for u in G}
else:
    raise Exception("Invalid scheme")

device_list = list(devices.values())
accuracies = list()
for epoch in range(100):
    messages = list()
    for u, v in tqdm(G.edges()):
        if random() <= 0.1:
            message = devices[u].send(devices[v])
            if scheme == "gossip":  # but not ngossip
                message = message[0:2] + (choice(device_list).send()[2:])
            messages.append(len(pickle.dumps(message)))
            message = devices[v].receive(devices[u], message)
            if scheme == "gossip": # but not ngossip
                message = message[0:2] + (choice(device_list).send()[2:])
            messages.append(len(pickle.dumps(message)))
            message = devices[u].ack(devices[v], message)
    accuracy = sum(1. if devices[u].predict() == labels[u] else 0 for u in test) / len(test)
    print("Epoch", epoch, "Accuracy", accuracy, "message size", sum(messages) / float(len(messages)))
    accuracies.append(accuracy)
print(dataset+"_"+scheme, accuracies, ";")

