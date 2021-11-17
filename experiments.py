import decentralized
import learning.nn
import numpy as np


def experiment(dataset,
               device_type=decentralized.devices.GossipDevice,
               gossip_merge=decentralized.mergers.AvgMerge,
               classifier=learning.nn.MLP,
               pretrained=False,
               gossip_pull=False,
               seed=0):
    measures = {"acc": list(), "base_acc": list()}
    network, test_labels = decentralized.simulation.create_network(dataset, device_type,
                                                                   classifier=classifier,
                                                                   pretrained=pretrained,
                                                                   gossip_merge=gossip_merge,
                                                                   gossip_pull=gossip_pull,
                                                                   seed=seed)
    for epoch in range(1500):
        network.round()
        accuracy_base = sum(1. if network.devices[u].predict(False) == label else 0 for u, label in test_labels.items()) / len(test_labels)
        accuracy = sum(1. if network.devices[u].predict() == label else 0 for u, label in test_labels.items()) / len(test_labels)
        measures["base_acc"].append(accuracy_base)
        measures["acc"].append(accuracy)
        if epoch % 1 == 0:
            print(f"Epoch {epoch} \t Acc {accuracy:.3f} \t Base acc {accuracy_base:.3f}")
    return measures


setting = {"dataset": "citeseer",
           "device_type": decentralized.devices.CorpusDevice,
           "gossip_merge": decentralized.mergers.AvgMerge,
           "classifier": learning.nn.MLP,
           "pretrained": False,
           "gossip_pull": False}
print(setting)

measures = {"acc": list(), "base_acc": list()}
for repetition in range(1):
    for measure, curve in experiment(**setting, seed=repetition).items():
        if measure in measures:
            measures[measure].append(curve[-1])
print("Average "+str({measure: np.mean(values) for measure, values in measures.items()}))
