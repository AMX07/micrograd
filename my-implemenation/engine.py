#We need spl data structures for containing neural networks, as nn are basically huge math exp

class Value:

    def __init__(self,data,_children=(), _op='', label=''): # op is operation?
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None 
        self._prev = set(_children) #initally a tuple, but now a set for efficency
        self._op = _op #opeartation that produced the node
        self.label = label #?

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other),'+')

        def _backward():

        
        return out
    




    