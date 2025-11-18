import numpy as np
from michigrad.engine import Value
from michigrad.nn import MLP
from michigrad.visualize import show_graph

# Usamos -1 y 1 para las salidas para facilitar la convergencia con tanh más adelante, 
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
ys = [Value(-1.0), Value(1.0), Value(1.0), Value(-1.0)]

# nin=2 (2 entradas), nouts=[2, 1] (capa oculta de 2, salida de 1)
# IMPORTANTE: En la versión original de nn.py, el MLP aplica ReLU por defecto en las capas ocultas
# Para hacerlo LINEAL pasamos nonlin a False

model = MLP(2, [2, 1], nonlin=False) 

for i in range(50):
    # 1. Forward
    ypred = [model(x) for x in xs]
    
    # 2. Loss
    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys))
    
    # 3. Zero grad
    model.zero_grad()
    
    # 4. Backward
    loss.backward()
    
    # 5. Update
    learning_rate = 0.1
    for p in model.parameters():
        p.data += -learning_rate * p.grad
    
    print(f"Paso {i+1} | Pérdida: {loss.data:.4f}")

print(f"Pérdida final: {loss.data:.4f}")
print("Predicciones finales:", [y.data for y in ypred])

show_graph(loss).render('grafo_xor_lineal', format='png')
