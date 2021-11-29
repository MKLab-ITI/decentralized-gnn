import numpy as np
import random


class AvgMerge:
    def __init__(self, value, is_training=False):
        self.value = value
        self.is_training = is_training

    def set(self, value):
        self.value = value

    def receive(self, _, value):
        self.value = (self.value + value)*0.5

    def get(self):
        return self.value

    def send(self):
        return self.value


class SlowMerge:
    def __init__(self, value, is_training=False):
        self.value = value
        self.is_training = is_training

    def set(self, value):
        self.value = value

    def receive(self, _, value):
        self.value = self.value * 0.9 + value * 0.1

    def get(self):
        return self.value

    def send(self):
        return self.value


class TopologicalMerge:
    def __init__(self, value, is_training=False):
        self.value = value
        self.neighbors = dict()

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def receive(self, neighbor, value):
        value, nns = value
        self.neighbors[neighbor] = 1
        self.value = self.value + 0.2*(value-self.value)/len(self.neighbors)**0.5/max(1, nns)**0.5

    def send(self):
        return self.value, len(self.neighbors)


class RandomMergeVariable:
    def __init__(self, value, is_training=False, dims=20):
        if dims is None:
            dims = value.size
        self.value = np.ones((1,1))*value if isinstance(value, float) else value
        self.training_id = np.random.random(size=dims)*float(is_training)
        self.personalization_value = value
        self.personalization_training_id = self.training_id
        self.neighbor_values = dict()
        self.neighbor_training = dict()
        self.neighbor_weights = dict()
        self.dims = dims
        self.is_training = is_training

    def set(self, value):
        self.value = value
        self.personalization_value = value

    def _sum(self, neighbor_values):
        ret = 0
        for v in neighbor_values:
            ret = ret + neighbor_values[v]*self.neighbor_weights[v]
        return ret

    def receive(self, neighbor, message):
        value, training_id = message
        if np.linalg.norm(training_id, 2) == 0:
            return
        self.neighbor_values[neighbor] = value
        self.neighbor_training[neighbor] = training_id
        if self.is_training:
            self.neighbor_training[self] = self.personalization_training_id if self.is_training else self.training_id
            self.neighbor_values[self] = self.personalization_value if self.is_training else self.value
            self.neighbor_weights[self] = 1
        if neighbor not in self.neighbor_weights:
            self.neighbor_weights[neighbor] = 1
        if len(self.neighbor_weights) <= 1:
            return
        #weight_sum = sum(abs(val) for val in self.neighbor_weights.values())
        self.neighbor_weights = {k: 1./len(self.neighbor_weights) for k, v in self.neighbor_weights.items()}
        error = [100000]
        for _ in range(1000):
            prev_error = error
            error = self._sum(self.neighbor_training) - 0.5
            for v in self.neighbor_weights:
                self.neighbor_weights[v] -= 0.01*np.sum(error*self.neighbor_training[v])/self.dims
            weight_sum = sum(abs(val) for val in self.neighbor_weights.values())
            self.neighbor_weights = {k: v/weight_sum for k, v in self.neighbor_weights.items()}
            if abs(np.linalg.norm(error)-np.linalg.norm(prev_error)) < 0.0001:
                break
        self.value = self._sum(self.neighbor_values)
        self.training_id = self._sum(self.neighbor_training)

    def get(self):
        return self.value

    def send(self):
        return self.value, self.training_id


class Smooth:
    def __init__(self, var):
        self.var = var
        self.value = 0
        self.beta = 0.9
        self.betat = 1

    def set(self, value):
        self.var.set(value)

    def get(self):
        if self.betat == 1:
            return self.var.get()
        return self.value / (1-self.betat)

    def receive(self, neighbor, value):
        self.var.receive(neighbor, value)
        self.update()

    def send(self):
        return self.var.send()

    def update(self):
        self.value = self.var.get()*(1-self.beta) + self.beta*self.value
        self.betat *= self.beta


class PPRVariable:
    def __init__(self, value, update_rule="PPR", balance=0.5, is_training=False):
        self.neighbors = dict()
        self.personalization = None
        self.balance = balance
        self.is_training = is_training
        if update_rule=="PPR":
            self.update_rule = lambda n,p: 0.9*n+0.1*p
        elif update_rule=="PR":
            self.update_rule = lambda n,p: n
        elif update_rule=="FDiff":
            self.update_rule = lambda n,p: n*0.9+0.1*p if not self.is_training else p
        elif update_rule=="AVG":
            self.update_rule = lambda n,p: (n*len(self.neighbors)**(1-self.balance)+p) / (len(self.neighbors)+1)**(1-self.balance)
        elif update_rule=="CHOCO":
            self.update_rule = lambda n,p: (p+(n*len(self.neighbors)**(1-self.balance)-p*len(self.neighbors))/len(self.neighbors)**(1-self.balance))
        else:
            self.update_rule = update_rule
        self.set(value)

    def set(self, value):
        self.neighbors[self] = value
        self.personalization = value
        self.update()

    def get(self):
        return self.neighbors[self]

    def receive(self, neighbor, value):
        self.neighbors[neighbor] = value
        self.update()

    def send(self):
        return self.get() / len(self.neighbors)**self.balance

    def update(self):
        aggregate = 0
        for value in self.neighbors.values():
            aggregate = aggregate + value
        #prev_value = self.neighbors.get(self, None)
        self.neighbors[self] = self.update_rule(aggregate/len(self.neighbors)**(1-self.balance), self.personalization)