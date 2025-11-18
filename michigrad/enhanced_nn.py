import random
from michigrad.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    """
    Neurona lineal: y = w * x + b
    """

    def __init__(self, nin):
        """
        nin: número de entradas
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        """
        x: lista de entradas - 
        retorna: salida de la neurona - 
        w * x + b
        """
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act

    def parameters(self):
        """
        retorna: lista de parámetros (pesos y sesgo)
        """
        return self.w + [self.b]

    def __repr__(self):
        return f"LinearNeuron({len(self.w)})"

class Linear(Module):
    """
    Capa lineal: múltiples neuronas lineales
    """
    
    def __init__(self, nin, nout):
        """
        nin: número de entradas
        nout: número de salidas (neuronas)
        """
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        x: lista de entradas
        retorna: lista de salidas de las neuronas
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        retorna: lista de parámetros de todas las neuronas
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"LinearLayer([{', '.join(str(n) for n in self.neurons)}])"

class ReLU(Module):
    """
    Función de activación ReLU
    """
    def __call__(self, x):
        """
        x: entrada (Value o lista de Values)
        retorna: ReLU aplicada a la entrada
        """
        # Si x es una lista (salida de una capa anterior), aplicamos relu a cada elemento
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"

class Tanh(Module):
    """
    Función de activación Tanh
    """

    def __call__(self, x):
        """
        x: entrada (Value o lista de Values)
        retorna: Tanh aplicada a la entrada
        """

        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        return x.tanh()

    def __repr__(self):
        return "Tanh()"

class Sigmoid(Module):
    """
    Función de activación Sigmoid
    """

    def __call__(self, x):
        """
        x: entrada (Value o lista de Values)
        retorna: Sigmoid aplicada a la entrada
        """
        
        if isinstance(x, list):
            return [xi.sigmoid() for xi in x]
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Sequential(Module):
    """
    Secuencia de capas apiladas, similar a nn.Sequential de PyTorch
    """

    def __init__(self, layers):
        """
        layers: lista de capas (Module)
        """
        self.layers = layers

    def __call__(self, x):
        """
        x: entrada (Value o lista de Values)
        retorna: salida después de pasar por todas las capas
        """

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        retorna: lista de todos los parámetros de todas las capas
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Sequential([{', '.join(str(layer) for layer in self.layers)}])"
    