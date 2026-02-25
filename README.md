# LeIA 2025

Este repositorio centraliza el desarrollo técnico y las implementaciones realizadas para la cátedra **Lógica e Inteligencia Artificial (LeIA 2025)** de la UNLP.

---

## Módulos Principales

### 1. Simulación de Agente Aspiradora

Contiene la implementación del ecosistema de la aspiradora basado en el libro de Russell & Norvig:

* **Entorno (`EntornoAspiradora`)**: Clase que gestiona los cuadrantes y su estado de suciedad.


* **Agente (`AgentePatrullaSimple`)**: Lógica de percepción y decisión (Aspirar/Mover).


* **Métrica**: Función `calcular_rendimiento` que puntúa la limpieza por paso de tiempo.



### 2. Michigrad (Librería `MyNN`)

Motor de diferenciación automática (Autograd) y redes neuronales desarrollado desde cero:

* **Clase** `Value**`: Soporta el grafo de operaciones y el cálculo de gradientes mediante `backpropagation`.
* **Activaciones**: Implementación de ReLU, Tanh y Sigmoide.
* **Resolución de XOR**:
    * Script `xor_punto_1.py`: Intento con modelo lineal (falla por falta de separabilidad).
    * Script `xor_punto_3.py`: Resolución exitosa mediante capas no-lineales.

---

## Ejecución

Para ejecutar la aspiradora:

```bash
python ejercicio_5_tp_7.py

```

Para entrenar el modelo XOR lineal:

```bash
python xor_punto_1.py

```


Para entrenar el modelo XOR no lineal:

```bash
python xor_punto_3.py

```

El bucle ejecuta: `Forward` -> `Loss` -> `Zero Grad` -> `Backward` -> `Update`.

## Integrantes

* Cestorame, Giuliana María
* Gerez, Marcos Mateo 
* Reale, Ezequiel Iván 

---
