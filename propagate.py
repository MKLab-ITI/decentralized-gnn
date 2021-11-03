import numpy as np
from random import choices

class MergeVariable:
    def __init__(self, value, is_training=None):
        self.value = value

    def set(self, value):
        self.value = value

    def receive(self, _, value):
        self.value = (self.value + value)*0.5

    def get(self):
        return self.value

    def send(self):
        return self.value

class OrthoMergeVariable:
    def __init__(self, value, is_training=None):
        self.value = value

    def set(self, value):
        self.value = value

    def receive(self, _, value):
        norm = np.linalg.norm(value, 2)
        if norm > 0:
            #    value = value - (value*self.value).sum()*self.value/norm
            self.value = (self.value + value/norm*np.linalg.norm(self.value, 2))*0.5

    def get(self):
        return self.value

    def send(self):
        return self.value


class RandomMergeVariable:
    def __init__(self, value, is_training=False, dims=20):
        if dims is None:
            dims = value.size
        self.value = value
        self.training_id = (np.random.random(size=dims)-0.5)*float(is_training)
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

    def receive(self, neighbor, message):
        value, training_id = message
        self.neighbor_values[neighbor] = value
        self.neighbor_training[neighbor] = training_id
        if neighbor not in self.neighbor_weights:
            self.neighbor_weights[neighbor] = 1
            weight_sum = sum(abs(val) for val in self.neighbor_weights.values())
            self.neighbor_weights = {k: v/weight_sum for k, v in self.neighbor_weights.items()}
        alpha = 0.5
        error = [100000]
        for _ in range(1000):
            prev_error = error
            error = 0
            for v in self.neighbor_training:
                error = error + self.neighbor_training[v]*self.neighbor_weights[v] #/ len(self.neighbor_weights)
            error = error * alpha + (1-alpha) * self.personalization_training_id
            for v in self.neighbor_weights:
                self.neighbor_weights[v] -= 0.01*np.sum(error*self.neighbor_training[v])/self.dims
            #self.neighbor_weights = {k: max(v, 0.1/len(self.neighbor_weights)) for k, v in self.neighbor_weights.items()}
            weight_sum = sum(abs(val) for val in self.neighbor_weights.values())
            self.neighbor_weights = {k: v/weight_sum for k, v in self.neighbor_weights.items()}
            #print(np.linalg.norm(error))
            if abs(np.linalg.norm(error)-np.linalg.norm(prev_error)) < 0.001:
                break
        #print("finished iters")
        self.value = 0
        self.training_id = 0
        for v in self.neighbor_values:
            self.value = self.value + self.neighbor_values[v]*self.neighbor_weights[v]
            self.training_id = self.training_id + self.neighbor_training[v]*self.neighbor_weights[v]
        self.value = self.value * alpha + (1-alpha) * self.personalization_value
        #self.personalization_value = self.value
        self.training_id = self.training_id * alpha + (1-alpha) * self.personalization_training_id
        #self.personalization_training_id = self.training_id

    def get(self):
        return self.value#(self.value - self.personalization_value)/0.9

    def send(self):
        return self.value, self.training_id


class TopoMergeVariable:
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


class DecentralizedVariable:
    def __init__(self, variable, base_class, is_training=False):
        self.variable = variable
        self.merger = base_class(variable.value, is_training=is_training)

    def send(self):
        self.merger.set(self.variable.value)
        return self.merger.send()

    def get(self):
        return self.value

    def receive(self, neighbor, value):
        self.merger.receive(neighbor, value)
        self.variable.value = self.merger.get()


class Device:
    def __init__(self):
        self.vars = list()

    def append(self, var):
        self.vars.append(var)
        return var

    def send(self, device=None):
        return [var.send() for var in self.vars]

    def receive(self, device, message):
        reply = self.send(None)
        self.ack(device, message)
        return reply

    def ack(self, device, message):
        for var, value in zip(self.vars, message):
            var.receive(device, value)
