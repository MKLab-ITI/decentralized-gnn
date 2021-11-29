import decentralized
import learning.nn
import numpy as np


def experiment(dataset,
               device_type=decentralized.devices.GossipDevice,
               gossip_merge=None,
               classifier=learning.nn.LR,
               pretrained=False,
               gossip_pull=False,
               smoothen=decentralized.mergers.Smooth,
               seed=0):
    measures = {"acc": list(), "base_acc": list()}
    network, test_labels = decentralized.simulation.create_network(dataset, device_type,
                                                                   classifier=classifier,
                                                                   pretrained=pretrained,
                                                                   gossip_merge=gossip_merge,
                                                                   gossip_pull=gossip_pull,
                                                                   smoothen=smoothen,
                                                                   seed=seed)
    for epoch in range(2000):
        network.round()
        accuracy_base = sum(1. if network.devices[u].predict(False) == label else 0 for u, label in test_labels.items()) / len(test_labels)
        accuracy = sum(1. if network.devices[u].predict() == label else 0 for u, label in test_labels.items()) / len(test_labels)
        measures["base_acc"].append(accuracy_base)
        measures["acc"].append(accuracy)
        if epoch % 1 == 0:
            print(f"Epoch {epoch} \t Acc {accuracy:.3f} \t Base acc {accuracy_base:.3f}")
    return measures


setting = {"dataset": "citeseer",
           "device_type": decentralized.devices.GossipDevice,
           "gossip_merge": None,
           "classifier": learning.nn.MLP,
           "pretrained": False,
           "gossip_pull": False}
print(setting)

measures = {"acc": list(), "base_acc": list()}
measures_curve_std = {measure: list() for measure in measures}
curves = {measure: list() for measure in measures}
for repetition in range(1):
    for measure, curve in experiment(**setting, seed=repetition).items():
        if measure in measures:
            measures[measure].append(curve[-1])
            measures_curve_std[measure].append(np.std(curve[-100:]))
            curves[measure].append(curve)
print("Average "+str({measure: [np.mean(values), np.std(values)] for measure, values in measures.items()}))
print("Average stds "+str({measure: [np.mean(values), np.std(values)] for measure, values in measures_curve_std.items()}))
curve1 = curves["acc"][0]
curve4 = curves["base_acc"][0]



setting = {"dataset": "citeseer",
           "device_type": decentralized.devices.GossipDevice,
           "gossip_merge": decentralized.mergers.AvgMerge,
           "classifier": learning.nn.MLP,
           "pretrained": False,
           "gossip_pull": False}
print(setting)

measures = {"acc": list(), "base_acc": list()}
curves = {measure: list() for measure in measures}
for repetition in range(1):
    for measure, curve in experiment(**setting, seed=repetition).items():
        if measure in measures:
            measures[measure].append(curve[-1])
            curves[measure].append(curve)
print("Average "+str({measure: [np.mean(values), np.std(values)] for measure, values in measures.items()}))
curve2 = curves["acc"][0]
curve3 = curves["base_acc"][0]

from matplotlib import pyplot as plt
plt.plot(curve2)
plt.plot(curve3)
#plt.plot(curve1)
#plt.plot(curve4)
plt.legend(["PullGossip p2pGNN", "PullGossip Base", "LabelDiff", "Random"])
plt.ylabel("Accuracy")
plt.ylabel("Time")
plt.show()