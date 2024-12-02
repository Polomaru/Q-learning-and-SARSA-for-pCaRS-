def parse_txt_file(file_path):
    # Inicialización de variables
    name = ""
    type_problem = ""
    comment = ""
    dimension = 0
    cars_number = 0
    edge_weight_type = ""
    edge_weight_format = ""
    edge_weight_matrix = []
    return_rate_matrix = []

    # Abrir el archivo
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterar sobre las líneas para extraer la información
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
def calcular_costo_total(nodo_origen, nodo_destino, carro_idx, edge_weight_matrix):
    return edge_weight_matrix[carro_idx][nodo_origen][nodo_destino]


def calcular_return_total(nodo_origen, nodo_destino, carro_idx, return_rate_matrix):
    return return_rate_matrix[carro_idx][nodo_origen][nodo_destino]


def calcular_ruta_completa(ruta, asignacion_carros, edge_weight_matrix, return_rate_matrix):
    costo_total = 0
    for i in range(1, len(ruta)):
        nodo_origen = ruta[i - 1]
        nodo_destino = ruta[i]
        carro_idx = asignacion_carros[i - 1]
        costo_total += calcular_costo_total(nodo_origen, nodo_destino, carro_idx, edge_weight_matrix)

    nodo_origen = ruta[-1]
    nodo_destino = ruta[0]
    carro_idx = asignacion_carros[-1]
    costo_total += calcular_costo_total(nodo_origen, nodo_destino, carro_idx, edge_weight_matrix)


    costadi = 0
    init = asignacion_carros[0]
    initx = ruta[0]
    for k in range(len(asignacion_carros)):
        if init != asignacion_carros[k]:
            costadi += calcular_return_total(initx, ruta[k], init, return_rate_matrix)
            init = asignacion_carros[k]
            initx = ruta[k]

    if initx != ruta[-1]:
        costadi += calcular_return_total(initx, ruta[-1], init, return_rate_matrix)

    return costo_total + costadi


def generar_rutas_iniciando_desde(nodo_inicio, n_vertices):
    nodos = list(range(n_vertices))
    nodos.remove(nodo_inicio)
    permutaciones_rutas = itertools.permutations(nodos)
    return [[nodo_inicio] + list(perm) + [nodo_inicio] for perm in permutaciones_rutas]


def asignar_carros_eficiente(n_carros, n_vertices):

    return itertools.product(range(n_carros), repeat=n_vertices)


def encontrar_mejor_ruta(nodo_inicio, n_carros, n_vertices, edge_weight_matrix, return_rate_matrix):
    rutas_posibles = generar_rutas_iniciando_desde(nodo_inicio, n_vertices)

    mejor_ruta, mejor_asignacion_carros, mejor_costo = None, None, float('inf')

    for ruta in rutas_posibles:
        for asignacion_carros in asignar_carros_eficiente(n_carros, n_vertices):
            costo_total = calcular_ruta_completa(ruta, asignacion_carros, edge_weight_matrix, return_rate_matrix)
            if costo_total < mejor_costo:
                mejor_costo = costo_total
                mejor_ruta = ruta
                mejor_asignacion_carros = asignacion_carros

    return mejor_ruta, mejor_asignacion_carros, mejor_costo



file_path = 'test/AfricaSul11n.pcar'
file_path = "temp.txt"
dimension, edge_weight_matrices, return_rate_matrices = parse_txt_file(file_path)

n_vertices = dimension
n_carros = len(edge_weight_matrices)
nodo_inicio = 0

mejor_ruta, mejor_asignacion_carros, mejor_costo = encontrar_mejor_ruta(nodo_inicio, n_carros, n_vertices, edge_weight_matrices, return_rate_matrices)

print(f"Mejor ruta: {mejor_ruta}")
print(f"Mejor asignación de carros: {mejor_asignacion_carros}")
print(f"Mejor costo total: {mejor_costo}")