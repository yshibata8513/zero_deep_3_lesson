
import numpy as np

def as_array(x):
    if not np.isscalar(x):
        return np.array(x)
    return x

def as_variable(x):
    if isinstance(x,Variable):
        return x
    return Variable(x)


class Function:
    def __call__(self,*xs):
        self.inputs = [as_variable(x) for x in xs]
        xs_ = [x.data for x in self.inputs]
        ys = self.forward(*xs_)

        if not isinstance(ys,tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for y in outputs:
            y.set_creator(self)
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self,*p):
        raise NotImplementedError

    def backward(self,gy):
        raise NotImplementedError

class Add(Function):
    def forward(self,x,y):
        return x + y
    def backward(self,gy):
        return np.ones(1),np.ones(1)

class Mul(Function):
    def forward(self,x,y):
        return x*y
    def backward(self,gy):
        x,y = self.inputs
        return y,x


def add(x,y):
    return Add()(x,y)


def mul(x,y):
    return Mul()(x,y)

class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func


    def backward(self):
        if self.grad==None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]

        gy = self.grad
        while(funcs):
            func = funcs.pop()
            print(func)
            gxs = func.backward(gy)
            for x,gx in zip(func.inputs,gxs):
                x.grad = gx
                if x.creator:
                    x.backward()
            

    def __mul__(self,other):
        return mul(self,other)

    def __add__(self,other):
        return add(self,other)

    def __rmul__(self,other):
        return mul(other,self)

    def __radd__(self,other):
        return add(other,self)


a = Variable(np.array(3))
b = Variable(np.array(2))
c= 3*(a + b)
print(c.data)
c.backward()
print(a.grad)