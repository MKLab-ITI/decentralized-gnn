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
