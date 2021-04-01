import numpy as np
import networkx as nx
import os.path
import pickle


def load(dataset, verbose=True, radius=None):
    if os.path.exists("data/"+dataset+".pickle"):
        ret = pickle.load(open("data/"+dataset+".pickle", "rb"))
    else:
        import dgl.data
        dataloader = {"cora": dgl.data.CoraGraphDataset, "citeseer": dgl.data.CiteseerGraphDataset, "pubmed": dgl.data.PubmedGraphDataset}
        data = dataloader[dataset](verbose=False)[0]
        G = nx.DiGraph(zip(data.edges()[0].numpy().tolist(), data.edges()[1].numpy().tolist()))
        features = dict(zip(data.nodes().numpy().tolist(), data.ndata['feat'].numpy().tolist()))
        labels = dict(zip(data.nodes().numpy().tolist(), data.ndata['label'].numpy().tolist()))
        training = np.where(data.ndata['train_mask'])[0].tolist()
        validation = np.where(data.ndata['val_mask'])[0].tolist()
        test = np.where(data.ndata['test_mask'])[0].tolist()
        ret = G, features, labels, set(training), set(validation), set(test)
        pickle.dump(ret, open("data/"+dataset+".pickle", "wb"))

    if radius is not None:
        G, features, labels, training, validation, test = ret
        G = nx.ego_graph(G, list(G)[1], radius=radius)
        nodes = set(G)
        ret = G, features, labels, set([u for u in training if u in nodes]), set([u for u in validation if u in nodes]), set([u for u in test if u in nodes])

    if verbose:
        print("===== Dataset =====")
        print("Name :", dataset)
        print("Nodes:", len(ret[0]))
        print("Edges:", ret[0].number_of_edges())
        print("Train:", len(ret[3]))
        print("Valid:", len(ret[4]))
        print("Test :", len(ret[5]))
    return ret
