class DecentralizedVariable:
    def __init__(self, value, update_rule="PPR", balance=0.5):
        self.neighbors = dict()
        self.value = None
        self.personalization = None
        self.balance = balance
        if update_rule=="PPR":
            self.update_rule = lambda n,p: 0.9*n+0.1*p
        elif update_rule=="FDiff":
            self.update_rule = lambda n,p: n if p.sum() == 0 else p
        else:
            self.update_rule = update_rule
        self.set(value)

    def set(self, value):
        self.neighbors[self] = value
        self.value = value
        self.personalization = value

    def receive(self, neighbor, value):
        self.neighbors[neighbor] = value

    def send(self):
        return self.value / len(self.neighbors)**self.balance

    def update(self):
        aggregate = 0
        for value in self.neighbors.values():
            aggregate = aggregate + value
        self.value = self.update_rule(aggregate/len(self.neighbors)**(1-self.balance), self.personalization)


class Device:
    def __init__(self):
        self.vars = list()

    def send(self, device=None):
        return [var.send() for var in self.vars]

    def receive(self, device, message):
        self.ack(device, message)
        return self.send(None)

    def ack(self, device, message):
        for var, value in zip(self.vars, message):
            var.receive(device, value)
            var.update()