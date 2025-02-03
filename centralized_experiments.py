from data import importer
from learning.predict import train_or_load_MLP, onehot
import learning.nn
import numpy as np

for dataset in ["citeseer", "cora", "pubmed"]:
    classifier = learning.nn.LR
    graph, features, labels, training, validation, test = importer.load(
        dataset, verbose=False
    )
    num_classes = len(set(labels.values()))
    num_features = len(list(features.values())[0])
    onehot_labels = {u: onehot(labels[u], num_classes) for u in graph}
    empty_label = onehot(None, num_classes)
    training = set(training)
    validation = set(validation)
    training_labels = {
        u: onehot_labels[u] if u in training or u in validation else empty_label
        for u in graph
    }
    # print(len(training), len(validation), len(test))
    is_training = {u: False for u in graph}
    for u in training:
        is_training[u] = True
    for u in validation:
        is_training[u] = True
    if classifier is not None:
        pretrained = train_or_load_MLP(
            dataset,
            features,
            onehot_labels,
            num_classes,
            training,
            validation,
            test,
            classifier=classifier,
        )
    for u, v in list(graph.edges()):
        graph.add_edge(v, u)
    test_labels = {u: labels[u] for u in test}

    predictions = (
        {u: pretrained(features[u]) for u in graph}
        if classifier is not None
        else {u: training_labels[u] for u in graph}
    )
    errors = {
        u: training_labels[u] - predictions[u] if is_training[u] else training_labels[u]
        for u in graph
    }
    diffused_errors = errors
    for round in range(200):
        next_diffused_errors = {u: 0 for u in graph}
        for u in graph:
            if is_training[u]:
                next_diffused_errors[u] = errors[u]
            else:
                for v in graph.neighbors(u):
                    next_diffused_errors[u] = next_diffused_errors[u] + diffused_errors[
                        v
                    ] / (graph.out_degree(v) + 1)
                next_diffused_errors[u] = next_diffused_errors[u] + diffused_errors[
                    u
                ] / (graph.out_degree(u) + 1)
        diffused_errors = next_diffused_errors

    # sigma = 0
    # for u in training:
    #    sigma = sigma + np.sum(np.abs(diffused_errors[u])) / len(training)

    # combined_predictions = {u: sigma*diffused_errors[u]/(np.sum(np.abs(diffused_errors[u]))+1.E-8) + predictions[u] if not is_training[u] else onehot_labels[u] for u in graph}
    combined_predictions = {
        u: (
            diffused_errors[u] + predictions[u]
            if not is_training[u]
            else onehot_labels[u]
        )
        for u in graph
    }
    diffused_predictions = combined_predictions
    for round in range(200):
        next_diffused_predictions = {u: 0 for u in graph}
        for u in graph:
            for v in graph.neighbors(u):
                next_diffused_predictions[u] = next_diffused_predictions[
                    u
                ] + diffused_predictions[v] / ((graph.out_degree(v) + 1) ** 0.5)
            next_diffused_predictions[u] = next_diffused_predictions[
                u
            ] + diffused_predictions[u] / ((graph.out_degree(u) + 1) ** 0.5)
            next_diffused_predictions[u] = (
                next_diffused_predictions[u] / ((graph.out_degree(u) + 1) ** 0.5) * 0.9
                + 0.1 * combined_predictions[u]
            )
        diffused_predictions = next_diffused_predictions

    accuracy = sum(
        1.0 if np.argmax(diffused_predictions[u]) == label else 0
        for u, label in test_labels.items()
    ) / len(test_labels)
    print(dataset, accuracy)
