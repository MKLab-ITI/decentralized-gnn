import os
import pickle
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid

def load(dataset, verbose=True, radius=None):
    data_path = f"data/{dataset}.pickle"

    if os.path.exists(data_path):
        ret = pickle.load(open(data_path, "rb"))
    else:
        dataset_loader = Planetoid(root="data", name=dataset)
        data = dataset_loader[0]

        G = nx.DiGraph(zip(data.edge_index[0].numpy().tolist(), data.edge_index[1].numpy().tolist()))
        features = dict(zip(range(data.x.shape[0]), data.x.numpy().tolist()))
        labels = dict(zip(range(data.y.shape[0]), data.y.numpy().tolist()))
        training = set(torch.where(data.train_mask)[0].numpy().tolist())
        validation = set(torch.where(data.val_mask)[0].numpy().tolist())
        test = set(torch.where(data.test_mask)[0].numpy().tolist())

        validation = [u for u in validation if u in features and u in labels and u in G]
        test = [u for u in test if u in features and u in labels and u in G]
        training = [u for u in training if u in features and u in labels and u in G]
        ret = G, features, labels, training, validation, test

        pickle.dump(ret, open(data_path, "wb"))

    if radius is not None:
        G, features, labels, training, validation, test = ret
        G = nx.ego_graph(G, list(G)[1], radius=radius)
        nodes = set(G)
        ret = G, features, labels, training & nodes, validation & nodes, test & nodes

    if verbose:
        print("===== Dataset =====")
        print("Name :", dataset)
        print("Nodes:", len(ret[0]))
        print("Edges:", ret[0].number_of_edges())
        print("Train:", len(ret[3]))
        print("Valid:", len(ret[4]))
        print("Test :", len(ret[5]))

    return ret
