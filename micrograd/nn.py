#engine.py is the machinery to build out pretty complicated mathematical expressions.
# nn are just a specific class of mathematical expressions

import random
# import Value from engine.py

class Neuron:
    # sum (xiwi + b)
    #nin how many inputs come to a neuron
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1)) # controls the trigger happiness of the neuron
    
    def __call__(self,x): #for calling n(x), when n = neuron(2 = dimenions= len(x)?), x = [2,3] input vectors/
        #wi*xi+b, dot product of w and x
        # sum up all  wi*xi+b
        act=sum((wi*xi for wi, xi in zip(self.w,x)),self.b)
        out = act.tanh() 
        return out
    #should nin == len(x)?
# x = [2,3]
# n = Neuron(2) gives 2 wi(s)
# n(x) = computes x0w0 + b and x1w1 + b

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] #nout num of neurons in 1 layer
        # creates a list of neurons with their wi and b


    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        #evals the out
        #[out1 out2]

#l = Layer(2-neurons dim, 3 number of neurons)
#l(x)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self,nin,nouts): #nouts list of  sizes of all the layers we want in a mlp
        sz = [nin] + nouts
        self.layers = [(Layer(sz[i]), sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    #parameters are just weights and biases (things we can adjust and tune w/ backprop to get better output)
    #we dont tune/change the xs

    
# x = [2,3,-1]
# n = MLP(3,[4,4,1])
# n(x), forward pass in mlp

# what happens here? 
'''
x = [2,3,-1] this is the input vector features

n = MLP(3,[4,4,1])

n is object of MLP class, it initalizes the MLP class which does the following:

creates len(nouts) layers (3 = len([4,4,1]))
these layers are filled with [4,4,1] number of neurons respectively
the dimension of these neurons are 3 (w1,w2,w3)
the MLP classes initializes the layer class for creating layers 
which are basically just list of neurons,
so neuron class is also initialized which basically just create random 3 weights and 1 biases 
for of the neurons in layer

next we actually call/ compute by doing the forward pass
n(x)
this calls the mlp
which passes the x inputs into the layers which then passes it into the individual neurons
which then calculate their output.

so this way we get some outputs for all the neurons in all the layers of the MLP



basic nn training:

forward pass

p.grad = 0 (initally already 0 but for future we need to reset it)

loss.backward()

update the parameters
for p in n.parameters():

p.data += -0.01 (step size) * p.grad 



 





'''

