# decentralized-gnn
A package for implementing and simulating decentralized Graph Neural Network algorithms
for classification of peer-to-peer nodes. Developed code supports the publication
*p2pGNN: A Decentralized Graph Neural Network for Node Classification in Peer-to-Peer Networks*.

# :zap: Quick Start
To generate a local instance of a decentralized learning device:
```python
from decentralized.devices import GossipDevice
from decentralized.mergers import SlowMerge
from learning.nn import MLP
node = ... # a node identifier object (can be any object)
features = ... # feature vector, should have the same length for each device
labels = ... # one hot encoding of class labels, zeroes if no label is known
predictor = MLP(features.shape[0], labels.shape[0])  # or load a pretrained model with
device = GossipDevice(node, predictor, features, labels, gossip_merge=SlowMerge)
```

In this code, the type of the device (`GossipDevice`)and the variable merge protocol 
(`SlowMerge`) work together to define a decentralized learning seting for 
a Graph Neural Network that runs on and takes account of unstructured peer-to-peer links
of uncertain availability.

Then, when possible (e.g. at worst, whenever devices send messages to the others for
other reasons) perform the following information exchange scheme between linked devices 
`u` and `v`:

```python
send = u.send()
receive = v.receive(u.name, send)
u.ack(v.name, receive)
```


## :hammer_and_wrench: Simulations
Simulations on many devices automatically generated by existing datasets 
can be easily set up and run per the following code:

```python
from decentralized.devices import GossipDevice
from decentralized.mergers import AvgMerge
from decentralized.simulation import create_network

dataset_name = ... # "cora", "citeseer" or "pubmed"
network, test_labels = create_network(dataset_name, 
                                      GossipDevice,
                                      pretrained=False,
                                      gossip_merge=AvgMerge,
                                      gossip_pull=False,
                                      seed=0)
for epoch in range(800):
    network.round()
    accuracy_base = sum(1. if network.devices[u].predict(False) == label else 0 for u, label in test_labels.items()) / len(test_labels)
    accuracy = sum(1. if network.devices[u].predict() == label else 0 for u, label in test_labels.items()) / len(test_labels)
    print(f"Epoch {epoch} \t Acc {accuracy:.3f} \t Base acc {accuracy_base:.3f}")
```

In the above code, datasets are automatically downloaded using DGL's interface.
Then, devices are instantiated given desired setting preferences.

:warning: Some merge schemes take up a lot of memory to simulate.

# :notebook: Citation
```
TBD
```