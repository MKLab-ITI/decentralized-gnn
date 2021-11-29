import random
from data import importer
from learning.predict import train_or_load_MLP, onehot
from learning.nn import MLP
import concurrent.futures
import threading
from tqdm import tqdm


def create_network(dataset, device_type, classifier=MLP, pretrained=False, seed=0, gossip_pull=False, **kwargs):
    graph, features, labels, training, validation, test = importer.load(dataset, verbose=False)
    #training, test = test, training
    num_classes = len(set(labels.values()))
    num_features = len(list(features.values())[0])
    onehot_labels = {u: onehot(labels[u], num_classes) for u in graph}
    empty_label = onehot(None, num_classes)
    training = set(training)
    validation = set(validation)
    training_labels = {u: onehot_labels[u] if u in training or u in validation else empty_label for u in graph}
    for u, v in list(graph.edges()):
        graph.add_edge(v, u)

    if pretrained:
        if "gossip_merge" not in kwargs:
            kwargs["gossip_merge"] = None
        pretrained = train_or_load_MLP(dataset, features, onehot_labels, num_classes, training, validation, test, classifier=classifier)

        def init_classifier(u):
            return pretrained
    else:

        def init_classifier(u):
            return classifier(num_features, num_classes)

    network = Network(graph,
                      lambda u: device_type(u, init_classifier(u), features[u], training_labels[u],
                                            train_steps=1 if u in training else 0, **kwargs),
                      seed=seed,
                      gossip_pull=gossip_pull)
    test_labels = {u: labels[u] for u in test}
    del graph
    del features
    del training
    del validation
    del test
    return network, test_labels


class Network:
    def __init__(self, graph, init_device, seed=0, gossip_pull=False):
        self.init_device = init_device
        random.seed(seed)
        self.neighbors = {u: {v: random.random()*0.1 for v in graph.neighbors(u) if u != v} for u in graph}
        self.devices = {u: init_device(u) for u in graph}
        self.gossip_pull = gossip_pull
        self.device_list = list(self.devices.values()) # values of self.devices stored as a list

    def _random_protocol(self, message):
        if self.gossip_pull:
            return message[0:2] + random.choice(self.device_list).send()[2:]
        return message

    def _communicate(self, u, v):
        send = self._random_protocol(self.devices[u].send(self.devices[v]))
        receive = self._random_protocol(self.devices[v].receive(self.devices[u], send))
        self.devices[u].ack(self.devices[v], receive)


    def round(self):
        experiment_edges = [(u,v) for u, neighbors in self.neighbors.items() for v, freq in neighbors.items() if random.random() <= freq]
        random.shuffle(experiment_edges)
        communicating = set()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for u, v in experiment_edges:
                if u not in communicating and v not in communicating:
                    communicating.add(u)
                    communicating.add(v)
                    executor.submit(self._communicate, u, v)
                    #self._communicate(u, v)
