from decentralized.mergers import RandomMergeVariable
import networkx as nx
import random
import numpy as np

random.seed(1)

G = nx.les_miserables_graph()
edges = list(G.edges())
mean = 0.1
std = 0.1

vars = {
    v: (
        RandomMergeVariable(mean - std + random.random() * std * 2, is_training=True)
        if random.random() < 0.2
        else RandomMergeVariable(0, is_training=False)
    )
    for v in G
}

true_mean = sum(vars[v].value for v in G if vars[v].is_training) / len(
    [vars[v] for v in G if vars[v].is_training]
)
print("True mean", true_mean)

means = list()
for epoch in range(5000):
    random.shuffle(edges)
    for u, v in edges:
        if random.random() < 0.1:
            messagev = vars[v].send()
            messageu = vars[u].send()
            vars[u].receive(v, messagev)
            vars[v].receive(u, messageu)
    vals = [vars[v].value for v in G if not vars[v].is_training and vars[v].value != 0]
    if len(vals) > 0:
        print(np.mean(vals), np.std(vals))
        means.append(np.mean(vals))
