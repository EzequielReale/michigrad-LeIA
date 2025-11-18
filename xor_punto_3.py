import random
from michigrad.engine import Value
from michigrad.enhanced_nn import Linear, Tanh, Sequential 
from michigrad.visualize import show_graph

# Entrada (2) -> Linear(2 neuronas) -> Tanh -> Linear(1 neurona) -> Tanh (opcional al final)
# Usamos Tanh porque es muy estable para XOR con targets -1 y 1.
model = Sequential([
    Linear(2, 2),
    Tanh(),
    Linear(2, 1),
    Tanh() 
])

# Tanh da salidas entre -1 y 1, así que ajustamos los targets a ese rango.
xs = [
    [Value(-1.0), Value(-1.0)],
    [Value(-1.0), Value(1.0)],
    [Value(1.0), Value(-1.0)],
    [Value(1.0), Value(1.0)],
]
ys = [Value(-1.0), Value(1.0), Value(1.0), Value(-1.0)] 

print("Entrenando XOR con MLP no-lineal...")
for k in range(100): # Esto lo voy a ir ajustando qsy
    
    # 1. Forward
    ypred = [model(x) for x in xs]
    
    # 2. Loss
    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys))
    
    # 3. Zero grad
    model.zero_grad()
    
    # 4. Backward
    loss.backward()
    
    # 5. Update
    lr = 0.1 
    for p in model.parameters():
        p.data += -lr * p.grad
    
    if k % 10 == 0:
        print(f"Step {k} | Loss: {loss.data:.4f}")

print(f"Final Loss: {loss.data:.4f}")
print("Predicciones finales:", [y.data for y in ypred])

show_graph(loss).render('grafo_xor_nonlineal', format='png')
