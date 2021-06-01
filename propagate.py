import numpy as np


class MergeVariable:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def receive(self, _, value):
        self.value = (self.value + value)*0.5

    def get(self):
        return self.value

    def send(self):
        return self.value


class TopoMergeVariable:
    def __init__(self, value):
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
    def __init__(self, value, update_rule="PPR", balance=0.5):
        self.neighbors = dict()
        self.personalization = None
        self.balance = balance
        if update_rule=="PPR":
            self.update_rule = lambda n,p: 0.9*n+0.1*p
        elif update_rule=="FDiff":
            self.update_rule = lambda n,p: n if np.sum(p) == 0 else p
        elif update_rule=="AVG":
            self.update_rule = lambda n,p: (n*len(self.neighbors)**(1-self.balance)+p) / (len(self.neighbors)+1)**(1-self.balance)
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
        self.neighbors[self] = self.update_rule(aggregate/len(self.neighbors)**(1-self.balance), self.personalization)


class DecentralizedVariable:
    def __init__(self, variable, base_class):
        self.variable = variable
        self.merger = base_class(variable.value)

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
        self.ack(device, message)
        return self.send(None)

    def ack(self, device, message):
        for var, value in zip(self.vars, message):
            var.receive(device, value)