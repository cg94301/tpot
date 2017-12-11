import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

class Thing(object):
    def __init__(self, name):
        self.name = name

obj = Thing('Awesome')

frozen = jsonpickle.encode(obj)

print frozen

thawed = jsonpickle.decode(frozen)

assert obj.name == thawed.name

s=bytearray('abcdef')
print s

frozen = jsonpickle.encode(s)
thawed = jsonpickle.decode(frozen)
print thawed
