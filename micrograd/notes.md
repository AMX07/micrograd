

1. define a Value node (data, grad)
    A NN is just a math exp, but we need spl data structures to store and compute these exps. Value class stores the data (in micrograd all data is basically just scalars, so just non-complex numbers), the parents and math operation involved in creating this node, accumalted gradient grad = d(output)/d(this), def of how to compute a grad for any node given the math operation.

    The reason we accumalte gradients is because a node can influence the final output/node in multiple ways, and according to calculas we should add (accumalte) all the gradients of different paths in which a node affects the final output.
    


2. build a computation graph via operator overloading
3. store parents + op for debugging
4. topological sort
5. backward pass via local derivatives + chain rule.



Derivative

A derivative measures how an output changes when its single input changes.

Partial derivative

A partial derivative measures how an output changes when one input changes while all other inputs are held constant.

Gradient

The gradient is the vector of partial derivatives, describing the local sensitivity of the output to each input variable independently.

Chain rule

The chain rule computes how a change in an upstream variable affects an output by summing the contributions from all paths through which that variable influences the output.


Derivative:
only for single variable calculas

1.
y = 2*x**3 + 4b + 3
dy/dx = 6x**2, if x = 2, 24  

(f(x+h) - f(x))/h, slope = rise/run

((2*(x+h)**3 + 4b + 3) - (2*x**3 + 4b + 3))/h 
= 2[ 3*x**2] = 6*x**2


2. z = x^2 + y^2

dz/dx = 2x if y doesnt depend on z. or  x and y are independent variables

partial derivative:
for multi-variable functions

∂z/∂x = 2x (when calculating partial derivates we assume other the variables is constant/fixed)

gradients:

A vector of partial derivatives of a function.




chain rule:


if we want to find dz/dx,
but we know dy/dx and dz/dy.

then dz/dx = dy/dx * dz/dy

accumaltion of gradients (+=):

node.grad = sum of (child.grad × local derivative)


b = a+a

self.grad += 1.0 * out.grad
other.grad += 1.0 * out.grad

