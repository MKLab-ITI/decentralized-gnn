from learning.nn import MLP
from gossip import onehot
import os
import pickle
import numpy as np


def train_or_load_MLP(name, features, onehot_labels, num_classes, training, validation, test):
    f = MLP(len(list(features.values())[0]), num_classes)
    if not os.path.exists('data/'+name+'model.pickle'):
        best_loss = float('inf')
        interest = 100
        for epoch in range(2000):
            for u in training:
                pred = f(features[u], is_training=True)
                f.backpropagate(onehot_labels[u])
            f.learner_end_batch()
            loss = -np.sum(np.log(f(features[u])+1.E-12)*onehot_labels[u]+(1-onehot_labels[u])*np.log(1+1.E-12-f(features[u])) for u in validation).sum()
            if loss < best_loss:
                best_vars = f.save()
                best_loss = loss
                interest = 100
                print("Epoch", epoch, loss,
                      "Validation Accuracy", sum(1. if np.argmax(f(features[u])) == np.argmax(onehot_labels[u]) else 0 for u in validation) / len(validation),
                      "Test Accuracy", sum(1. if np.argmax(f(features[u])) == np.argmax(onehot_labels[u]) else 0 for u in test) / len(test))
            else:
                interest -= 1
                if interest == 0:
                    break
        with open('mlp.pickle', 'wb') as file:
            pickle.dump(best_vars, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('mlp.pickle', 'rb') as file:
        best_vars = pickle.load(file)
        f.load(best_vars)
        print("Test Accuracy", sum(1. if np.argmax(f(features[u])) == np.argmax(onehot_labels[u]) else 0 for u in test)/len(test))

    return f