import random

class EntornoAspiradora:
    """
    Simula un entorno de N cuadrantes en fila.
    Los cuadrantes limpios pueden ensuciarse solos.
    """
    def __init__(self, num_cuadrantes, estado_inicial=None):
        self.num_cuadrantes = num_cuadrantes
        if estado_inicial:
            self.estado = estado_inicial
        else:
            self.estado = {i: 'Sucio' for i in range(num_cuadrantes)}
        
        self.ubicacion_agente = random.randint(0, num_cuadrantes - 1)

    def get_percepcion(self):
        estado_actual_cuadrante = self.estado[self.ubicacion_agente]
        return (self.ubicacion_agente, estado_actual_cuadrante)

    def actualizar_estado_agente(self, accion):
        """Actualiza el entorno basado en la acción del AGENTE."""
        if accion == 'Aspirar':
            self.estado[self.ubicacion_agente] = 'Limpio'
        elif accion == 'MoverDerecha':
            if self.ubicacion_agente < self.num_cuadrantes - 1:
                self.ubicacion_agente += 1
        elif accion == 'MoverIzquierda':
            if self.ubicacion_agente > 0:
                self.ubicacion_agente -= 1

    def paso_de_mundo(self, prob_ensuciar):
        """
        Simula la evolución del MUNDO.
        Cada cuadrante LIMPIO tiene una 'prob_ensuciar' de volverse 'Sucio'.
        """
        cuadrantes_ensuciados = []
        for i in range(self.num_cuadrantes):
            if self.estado[i] == 'Limpio':
                if random.random() < prob_ensuciar:
                    self.estado[i] = 'Sucio'
                    cuadrantes_ensuciados.append(i)
        return cuadrantes_ensuciados

    def get_estado_actual(self):
        return self.estado.copy()

class AgentePatrullaSimple:
    """
    Implementa un agente reactivo simple con estado interno (dirección).
    Es racional para un entorno en fila que se ensucia aleatoriamente.
    """
    def __init__(self, num_cuadrantes):
        self.num_cuadrantes = num_cuadrantes
        self.direccion = 1 # 1 para Derecha, -1 para Izquierda

    def actuar(self, percepcion):
        ubicacion, estado_cuadrado = percepcion
        
        if estado_cuadrado == 'Sucio':
            return 'Aspirar'
        
        if ubicacion == 0:
            self.direccion = 1
        elif ubicacion == self.num_cuadrantes - 1:
            self.direccion = -1
            
        if self.direccion == 1:
            return 'MoverDerecha'
        else:
            return 'MoverIzquierda'

def simular(entorno, agente, pasos_tiempo, prob_ensuciar):
    """
    Ejecuta la simulación principal.
    (+10 limpiar, -1 mover).
    """
    puntos_totales = 0
    historial_para_medida = []

    print("--- INICIO SIMULACIÓN (Entorno Estocástico) ---")
    print(f"Estado inicial: {entorno.estado}, Ubicación inicial: {entorno.ubicacion_agente}")
    print(f"Prob. de ensuciarse: {prob_ensuciar}\n")
    
    for i in range(pasos_tiempo):
        ensuciados = entorno.paso_de_mundo(prob_ensuciar)
        
        # Guardamos el estado para la medida de rendimiento original
        estado_actual = entorno.get_estado_actual()
        historial_para_medida.append(estado_actual)

        # El AGENTE percibe y actúa
        percepcion = entorno.get_percepcion()
        accion = agente.actuar(percepcion)
        
        entorno.actualizar_estado_agente(accion)
        
        # Calculamos puntos
        puntos_paso = 0
        if accion == 'Aspirar':
            puntos_paso = 10
        elif 'Mover' in accion:
            puntos_paso = -1
        puntos_totales += puntos_paso
        
        print(f"Paso {i}:")
        if ensuciados:
            print(f"  MUNDO: Se ensuciaron los cuadrantes {ensuciados}")
        print(f"  Estado (inicio paso): {estado_actual}")
        print(f"  Percepción: {percepcion} -> Acción: {accion} -> Puntos: {puntos_paso}")
        print(f"  Estado (fin paso): {entorno.estado}, Nueva Ubic: {entorno.ubicacion_agente}")
        
    print("\n--- FIN SIMULACIÓN ---")
    return puntos_totales, historial_para_medida


if __name__ == "__main__":
    PASOS_TIEMPO = 30
    NUM_CUADRANTES = 4
    # Probabilidad de que un cuadrante limpio se ensucie en CADA paso
    PROB_ENSUCIAR = 0.1 

    estado_inicial_sucio = {i: 'Sucio' for i in range(NUM_CUADRANTES)}
    
    entorno = EntornoAspiradora(
        num_cuadrantes=NUM_CUADRANTES, 
        estado_inicial=estado_inicial_sucio
    )
    entorno.ubicacion_agente = 0
    
    agente = AgentePatrullaSimple(num_cuadrantes=NUM_CUADRANTES)

    puntos_finales, historial = simular(entorno, agente, PASOS_TIEMPO, PROB_ENSUCIAR)

    print("\n--- CÁLCULO DE RENDIMIENTO ---")
    print(f"Puntos Totales (Acciones): {puntos_finales}")
    print(f"Pasos Totales: {PASOS_TIEMPO}")
    if PASOS_TIEMPO > 0:
        print(f"Rendimiento (Puntos/Paso): {puntos_finales / PASOS_TIEMPO:.2f}")

    puntos = 0
    for estado in historial:
        puntos += sum(1 for c in estado.values() if c == 'Limpio')
    
    print("\n--- CÁLCULO DE RENDIMIENTO ---")
    print(f"Puntos Totales (Estado): {puntos}")
    print(f"Puntuación Máxima Posible: {NUM_CUADRANTES * PASOS_TIEMPO}")
    if PASOS_TIEMPO > 0:
        print(f"Rendimiento (Limpieza promedio): {puntos / (NUM_CUADRANTES * PASOS_TIEMPO):.2%}")

