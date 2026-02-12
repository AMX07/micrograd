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
            # gradient using chain rule = summation of local gradient + global gradient.
            #dL/de we want
            # we know, d = e + c, so dd/de = 1 (local derivative always 1 in addition ) and dL/dd = -2 (out.grad)
            # therefore dL/de = 1 * -2 or self.grad = local derivative * child derivative = 1 * out.grad
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        
        def _backward():
            #we want dL/da
            # we know e = a*b, de/da = b and dL/de = out.grad (-2)
            # dL/da = other.data * out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

    #topological sort, 
    def backward(self):
        topo =[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                



            

        
        




    