def parse_txt_file(file_path):

    name = ""
    type_problem = ""
    comment = ""
    dimension = 0
    cars_number = 0
    edge_weight_type = ""
    edge_weight_format = ""
    edge_weight_matrix = []
    return_rate_matrix = []


    with open(file_path, 'r') as file:
        lines = file.readlines()


    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("TYPE"):
            type_problem = line.split(":")[1].strip()
        elif line.startswith("COMMENT"):
            comment = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("CARS_NUMBER"):
            cars_number = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            i += 1  

            temp = []
            while i < len(lines) and not lines[i].startswith("RETURN_RATE_SECTION"):
                row = list(map(int, lines[i].strip().split()))
                if(len(row) == 1): 
                    if len(temp) > 0:
                        edge_weight_matrix.append(temp)
                        temp = []
                else:
                    temp.append(row)
                i += 1
            edge_weight_matrix.append(temp)
            continue  
        elif line.startswith("RETURN_RATE_SECTION"):
            i += 1  

            temp = []
            while i < len(lines) and lines[i].strip():
                row = list(map(int, lines[i].strip().split()))
                if(len(row) == 1): 
                    if len(temp) > 0:
                        return_rate_matrix.append(temp)
                        temp = []
                else:
                    temp.append(row)
                i += 1
            return_rate_matrix.append(temp)
            continue  

        i += 1

    return dimension, edge_weight_matrix, return_rate_matrix

import numpy as np
import itertools
import random

file_path = 'test/Egito9n.pcar' 

dimension, edge_weight_matrices, return_rate_matrices = parse_txt_file(file_path)

import numpy as np
import random
import itertools
from itertools import product


n_vertices = dimension
n_carros = len(edge_weight_matrices)
n_iteraciones = 4000  # Número de iteraciones de Q-learning
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 0.2  # Probabilidad de exploración más alta para asignación de carros


nodo_inicial = 0

Q = np.random.uniform(low=-1, high=1, size=(n_vertices, n_carros))


def calcular_costo_total(nodo_origen, nodo_destino, carro_idx):
    return edge_weight_matrices[carro_idx][nodo_origen][nodo_destino]

def calcular_return_total(nodo_origen, nodo_destino, carro_idx):
    return return_rate_matrices[carro_idx][nodo_origen][nodo_destino]

def calcular_ruta_completa(ruta, asignacion_carros):
    costo_total = 0
    for i in range(1, len(ruta)):
        nodo_origen = ruta[i - 1]
        nodo_destino = ruta[i]
        carro_idx = asignacion_carros[i - 1]
        costo_total += calcular_costo_total(nodo_origen, nodo_destino, carro_idx)

    nodo_origen = ruta[-1]
    nodo_destino = ruta[0]
    carro_idx = asignacion_carros[-1]
    costo_total += calcular_costo_total(nodo_origen, nodo_destino, carro_idx)

    costadi = 0
    init = asignacion_carros[0]
    initx = ruta[0]
    for k in range(len(asignacion_carros)):
        if init != asignacion_carros[k]:
            costadi += calcular_return_total(initx, ruta[k], init)
            init = asignacion_carros[k]
            initx = ruta[k]

    if initx != ruta[-1]:
        costadi += calcular_return_total(initx, ruta[-1], init)

    return costo_total + costadi

def asignar_carros_eficiente(n_carros, n_vertices):
    return list(product(range(n_carros), repeat=n_vertices))

def q_learning():
    estado = nodo_inicial
    asignacion_carros = [-1] * n_vertices  
    visitados = set([nodo_inicial])

    for _ in range(n_iteraciones):
        if random.uniform(0, 1) < epsilon:

            carro_disponible = list(range(n_carros))
            random.shuffle(carro_disponible)  
            for i in range(n_vertices):
                if asignacion_carros[i] == -1:  
                    for carro in carro_disponible:
                        if carro not in asignacion_carros:  
                            asignacion_carros[i] = carro
                            break
        else:

            for i in range(n_vertices):
                if asignacion_carros[i] == -1:
                    asignacion_carros[i] = np.argmax(Q[i])  
        costo = 0
        for i in range(1, len(asignacion_carros)):
            costo += calcular_costo_total(i - 1, i, asignacion_carros[i - 1])

        for i in range(1, len(asignacion_carros)):
            if asignacion_carros[i] != asignacion_carros[i - 1]:
                costo += calcular_return_total(i - 1, i, asignacion_carros[i - 1])

        recompensa = -costo  

        Q[estado, asignacion_carros[estado]] = Q[estado, asignacion_carros[estado]] + alpha * (recompensa + gamma * np.max(Q[estado]) - Q[estado, asignacion_carros[estado]])

        siguiente_ciudad = None
        for i in range(n_vertices):
            if i not in visitados:
                siguiente_ciudad = i
                visitados.add(i)
                break

        if siguiente_ciudad is None:
            break

        estado = siguiente_ciudad

    mejor_ruta = [nodo_inicial]
    mejor_asignacion_carros = [-1] * n_vertices
    mejor_costo = float('inf')


    for ruta in itertools.permutations(range(n_vertices)):
        if ruta[0] != nodo_inicial:
            continue

        asignacion_carros = [-1] * n_vertices
        for i in range(n_vertices):
            if asignacion_carros[i] == -1:
                asignacion_carros[i] = np.argmax(Q[i])  

        costo_total = calcular_ruta_completa(ruta, asignacion_carros)

        if costo_total < mejor_costo:
            mejor_costo = costo_total
            mejor_ruta = ruta + (ruta[0],)
            mejor_asignacion_carros = list(asignacion_carros)

    mejor_ruta = [nodo_inicial] + list(mejor_ruta[1:-1]) + [nodo_inicial]
    mejor_asignacion_carros = mejor_asignacion_carros[:n_vertices]

    costo_final = calcular_ruta_completa(mejor_ruta, mejor_asignacion_carros)

    return mejor_ruta, mejor_asignacion_carros, costo_final

import time

# Inicializar variables para calcular promedios
tiempos = []
costos = []
mejor_ruta = None
mejor_asignacion_carros = None
mejor_costo = float('inf')

# Función para realizar los 10 testeos
for i in range(10):
    start_time = time.time()  # Marca el inicio del tiempo de ejecución
    
    # Realizar la llamada a la función q_learning
    ruta, asignacion_carros, costo_total = q_learning()

    end_time = time.time()  # Marca el final del tiempo de ejecución

    # Calcular el tiempo de ejecución
    tiempo_ejecucion = end_time - start_time
    tiempos.append(tiempo_ejecucion)
    costos.append(costo_total)

    # Si el costo es el más bajo, guardamos la mejor ruta y asignación
    if costo_total < mejor_costo:
        mejor_costo = costo_total
        mejor_ruta = ruta
        mejor_asignacion_carros = asignacion_carros

    # Imprimir solo el tiempo de ejecución de cada test
    print(f"Tiempo de ejecución del test {i + 1}: {tiempo_ejecucion:.4f} segundos")

# Calcular los promedios
promedio_tiempo = sum(tiempos) / len(tiempos)
promedio_costo = sum(costos) / len(costos)

# Mostrar resultados finales
print("\nResultados finales:")
print(f"Promedio del tiempo de ejecución: {promedio_tiempo:.4f} segundos")
print(f"Promedio del costo total: {promedio_costo:.4f}")
print(f"Mejor ruta con el costo más bajo: {mejor_ruta}")
print(f"Mejor asignación de carros: {mejor_asignacion_carros}")
