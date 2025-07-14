import numpy as np


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
    bonus_satisfaction = []
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
                if len(row) == 1:
                    if temp:
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
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("BONUS_SATISFACTION_SECTION"):
                row = list(map(int, lines[i].strip().split()))
                if len(row) == 1:
                    if temp:
                        return_rate_matrix.append(temp)
                        temp = []
                else:
                    temp.append(row)
                i += 1
            return_rate_matrix.append(temp)
            continue
        elif line.startswith("BONUS_SATISFACTION_SECTION"):
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("EOF"):
                bonus_satisfaction.extend(map(int, lines[i].strip().split()))
                i += 1
            continue
        i += 1
    return name, dimension, edge_weight_matrix, return_rate_matrix, bonus_satisfaction, cars_number
file_path = 'test/AfricaSul11n.pcar'
name, dimension, edge_weight_matrix, return_rate_matrix, bonus_satisfaction, cars_number = parse_txt_file(file_path)

def calcular_costo_total(carro_idx, nodo_origen, nodo_destino):
    global edge_weight_matrix
    return edge_weight_matrix[carro_idx][nodo_origen][nodo_destino]

def calcular_return_total(carro_idx, nodo_origen, nodo_destino):
    global return_rate_matrix
    return return_rate_matrix[carro_idx][nodo_destino][nodo_origen]

def calcular_ruta_completa(ruta, asignacion_carros):
    global edge_weight_matrix, return_rate_matrix
    costo_total = 0
    for i in range(1, len(ruta)):
        nodo_origen = ruta[i - 1]
        nodo_destino = ruta[i]
        carro_idx = asignacion_carros[i - 1]
        costo_total += calcular_costo_total(carro_idx, nodo_origen, nodo_destino)

    nodo_origen = ruta[-1]
    nodo_destino = ruta[0]
    carro_idx = asignacion_carros[-1]
    costo_total += calcular_costo_total(carro_idx, nodo_origen, nodo_destino)


    costadi = 0
    init = asignacion_carros[0]
    initx = ruta[0]

    for k in range(len(asignacion_carros)):
        if init != asignacion_carros[k]:
            costadi += calcular_return_total(init, initx, ruta[k] )
            init = asignacion_carros[k]
            initx = ruta[k]

    if initx != ruta[-1]:
        costadi += calcular_return_total(init, initx ,ruta[-1], )

    return [costo_total , costadi]

def compute_value_of_q_table(Q_table, start_node):
    greedy_route, greedy_cars = compute_greedy_route(Q_table,start_node)
    return calcular_ruta_completa(greedy_route, greedy_cars)

def lastcomput(Q_table, start_node):
    greedy_route, greedy_cars = compute_greedy_route(Q_table,start_node)
    #print(greedy_route)
    #print(greedy_cars)
    return calcular_ruta_completa(greedy_route, greedy_cars), greedy_route, greedy_cars

Q_table = np.zeros((cars_number, int(dimension), int(dimension)))


def compute_greedy_route(Q_table, start_node):

    #print(Q_table)
    num_cars = cars_number
    num_nodes = dimension
    
    route = []
    cars = []
    visited = np.ones(num_nodes, dtype=np.bool_)

    # Umbral y estado de bonificación
    limitbonus = getbonus()
    currentbonus = bonus_satisfaction[start_node]

    # Inicialización
    current_node = start_node
    current_car = -1
    route.append(current_node)
    visited[current_node] = False
    step = 1

    while currentbonus < limitbonus:
        best_q = -np.inf
        best_node = -1
        best_car = -1

        for car in range(num_cars):
            if car in cars and car!=current_car:
                continue
            for node in range(num_nodes):
                if not visited[node]:
                    continue
                q_val = Q_table[car, current_node, node]
                if q_val > best_q:
                    best_q = q_val
                    best_node = node
                    best_car = car

        current_node = best_node
        current_car = best_car
        currentbonus += bonus_satisfaction[current_node]
        route.append(current_node)
        cars.append(current_car)
        visited[current_node] = False

        if currentbonus >= limitbonus:
            break
        step += 1

    best_return_q = -np.inf
    best_return_car = current_car
    for car in range(num_cars):
        if car in cars and car!=current_car:
                continue
        q_val = Q_table[car, current_node, start_node]
        if q_val > best_return_q:
            best_return_q = q_val
            best_return_car = car


    route.append(start_node)
    cars.append(best_return_car)

    return route, cars

EPOCHS = 3000
LEARNING_RATE = 0.2
GAMMA = 0.95
EPSILON = 0.2



def currmax(Q_table, currentcar, row, mask, maskcars):

    best_node = -1
    best_car = currentcar
    max_q = -np.inf
    cars_number, _, dimension = Q_table.shape

    for car in range(cars_number):
        if not maskcars[car]:
            continue  # Este auto ya fue usado
        for to_node in range(dimension):
            if not mask[to_node]:
                continue  # Nodo ya visitado
            q_val = Q_table[car, row, to_node]
            if q_val > max_q:
                max_q = q_val
                best_node = to_node
                best_car = car

    return best_node, best_car, max_q

def currmaxcar(Q_table, from_node, to_node, maskcars):
    best_car = -1
    best_q = -np.inf
    for car in range(Q_table.shape[0]):
        if not maskcars[car]:
            continue
        q_val = Q_table[car, from_node, to_node]
        if q_val > best_q:
            best_q = q_val
            best_car = car
    return best_car, best_q


def getbonus():
    total_sum = sum(bonus_satisfaction)
    threshold = total_sum * 0.8

    return threshold

def eps_greedy_update(nodoinit, carinit, Q_table, mask, maskcars, route, cars):
    global LEARNING_RATE, GAMMA, EPSILON, dimension, edge_weight_matrix, return_rate_matrix, bonus_satisfaction

    # Marca el nodo y el coche inicial como visitados
    mask[nodoinit] = False
    maskcars[carinit] = False
    next_visit = 0
    reward = 0
    route.append(nodoinit)

    current = nodoinit
    currentcar = carinit
    pointchangecar = nodoinit

    # Determina cuánta bonificación podemos acumular antes de cerrar la ruta
    limitbonus = getbonus()
    currentbonus = bonus_satisfaction[nodoinit]

    for i in range(1, dimension):
        # Reconstruye listas de nodos y coches disponibles
        possible = np.where(mask == True)[0]
        possiblecars = np.where(maskcars == True)[0]

        # ========== Caso A ==========
        # Sólo queda un nodo, no hay coches libres y aún podemos acumular bonificación
        if len(possible) == 1 and len(possiblecars) == 0 and limitbonus > currentbonus:
            next_visit = possible[0]
            currentbonus += bonus_satisfaction[next_visit]
            # Recompensa: costo negativo del tramo
            reward = -calcular_costo_total(currentcar, current, next_visit)
            # Estimación de la siguiente acción: volver al inicio
            max_next = -calcular_costo_total(currentcar, next_visit, nodoinit)
            # Actualiza Q
            Q_table[currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[currentcar, current, next_visit])
            # Avanza
            current = next_visit
            mask[current] = False
            route.extend([next_visit, nodoinit])
            cars.extend([next_car, next_car])

        # ========== Caso B ==========
        # Sólo queda un nodo, hay coches libres y todavía falta bonificación
        elif len(possible) == 1 and limitbonus > currentbonus:
            next_visit = possible[0]
            currentbonus += bonus_satisfaction[next_visit]
            # Elige coche: exploración o explotación
            if np.random.random() < EPSILON:
                # Exploración: cambia a un coche aleatorio disponible
                maskcars[currentcar] = True
                next_car = np.random.choice(np.where(maskcars == True)[0])
            else:
                # Explotación: elige el coche con mayor Q para este salto
                maskcars[currentcar] = True
                next_car, _ = currmaxcar(Q_table, current, next_visit, maskcars)
            # Actualiza en función de si cambia o no de coche
            if next_car != currentcar:
                # ========== Caso B1 ==========
                # Cambio de coche antes de visitar el nodo
                reward = -calcular_costo_total(next_car, current, next_visit) - calcular_return_total(currentcar, pointchangecar, next_visit)
                max_next = -calcular_costo_total(next_car, next_visit, nodoinit)
                Q_table[next_car, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[next_car, current, next_visit])
                current = next_visit
                pointchangecar = next_visit
                currentcar = next_car
                mask[current] = False
                route.extend([next_visit, nodoinit])
                cars.extend([next_car, next_car])
            else:
                # ========== Caso B2 ==========
                # Permanece en el mismo coche para visitar el nodo
                reward = -calcular_costo_total(currentcar, current, next_visit)
                max_next = -calcular_costo_total(currentcar, next_visit, nodoinit)
                Q_table[currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[currentcar, current, next_visit])
                current = next_visit
                mask[current] = False
                route.extend([next_visit, nodoinit])
                cars.extend([next_car, next_car])

        # ========== Caso C ==========
        # Hay nodos y no hay coches libres, pero aún falta bonificación
        elif limitbonus > currentbonus and len(possiblecars) == 0:
            # Elige siguiente nodo: exploración o explotación
            if np.random.random() < EPSILON:
                next_visit = np.random.choice(possible)
            else:
                maskcars[currentcar] = True
                next_visit, _, _ = currmax(Q_table, currentcar, current, mask, maskcars)
                maskcars[currentcar] = False
            currentbonus += bonus_satisfaction[next_visit]
            reward = -calcular_costo_total(currentcar, current, next_visit)
            if limitbonus <= currentbonus:
                # ========== Caso C1 ==========
                # Se alcanza la bonificación al visitar este nodo y se cierra ruta
                max_next = -calcular_costo_total(currentcar, next_visit, nodoinit)
                Q_table[currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[currentcar, current, next_visit])
                # Append regreso a inicio
                route.extend([next_visit, nodoinit])
                cars.extend([currentcar, currentcar])
                # Q actualización del regreso
                next_reward = -calcular_costo_total(currentcar, next_visit, nodoinit) - calcular_return_total(currentcar, pointchangecar, nodoinit)
                Q_table[currentcar, next_visit, nodoinit] += LEARNING_RATE * (next_reward - Q_table[currentcar, next_visit, nodoinit])
                return Q_table
            else:
                # ========== Caso C2 ==========
                # Sigue recorriendo tras visitar nodo
                maskcars[currentcar] = True
                _, _, max_next = currmax(Q_table, currentcar, next_visit, mask, maskcars)
                maskcars[currentcar] = False
                Q_table[currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[currentcar, current, next_visit])
                current = next_visit
                mask[current] = False
                route.append(next_visit)
                cars.append(currentcar)

        # ========== Caso D ==========
        # Quedan nodos y coches libres, y la bonificación aún no se alcanza
        elif limitbonus > currentbonus:
            if np.random.random() < EPSILON:
                # Exploración: elige nodo y coche aleatorios
                next_visit = np.random.choice(possible)
                maskcars[currentcar] = True
                next_car = np.random.choice(np.where(maskcars == True)[0])
            else:
                # Explotación: elige par (nodo, coche) con mayor Q
                maskcars[currentcar] = True
                next_visit, next_car, _ = currmax(Q_table, currentcar, current, mask, maskcars)
                maskcars[currentcar] = False
            currentbonus += bonus_satisfaction[next_visit]
            if currentbonus >= limitbonus:
                # ========== Caso D1 ==========
                if next_car == currentcar:
                    # D1a: mismo coche, cierra tras visita
                    reward = -calcular_costo_total(currentcar, current, next_visit)
                else:
                    # D1b: cambia de coche justo antes de cerrar
                    reward = -calcular_costo_total(next_car, current, next_visit) - calcular_return_total(currentcar, pointchangecar, next_visit)
                
                max_next = -calcular_costo_total(next_car if next_car!=currentcar else currentcar, next_visit, nodoinit)
        
                Q_table[next_car if next_car!=currentcar else currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[next_car if next_car!=currentcar else currentcar, current, next_visit])
                # Cierre de ruta
                route.extend([next_visit, nodoinit])
                cars.extend([next_car, next_car])
                # Q retorno
                ret_reward = -calcular_costo_total(next_car, next_visit, nodoinit) - calcular_return_total(next_car, pointchangecar, nodoinit)
                Q_table[next_car, next_visit, nodoinit] += LEARNING_RATE * (ret_reward - Q_table[next_car, next_visit, nodoinit])
                return Q_table
            else:
                # ========== Caso D2 ==========
                # Continúa ruta con el par seleccionado
                if next_car == currentcar:
                    # D2a: mismo coche
                    reward = -calcular_costo_total(currentcar, current, next_visit)
                    maskcars[currentcar] = True
                    _, _, max_next = currmax(Q_table, currentcar, next_visit, mask, maskcars)
                    maskcars[currentcar] = False
                else:
                    # D2b: cambia de coche y sigue
                    reward = -calcular_costo_total(next_car, current, next_visit) - calcular_return_total(currentcar, pointchangecar, next_visit)
                    maskcars[next_car] = True
                    _, _, max_next = currmax(Q_table, next_car, next_visit, mask, maskcars)
                    maskcars[next_car] = False
                Q_table[next_car if next_car!=currentcar else currentcar, current, next_visit] += LEARNING_RATE * (reward + GAMMA * max_next - Q_table[next_car if next_car!=currentcar else currentcar, current, next_visit])
                current = next_visit
                pointchangecar = next_visit
                mask[current] = False
                route.append(next_visit)
                cars.append(next_car)
    return Q_table





def solving_qlearning(Q_table):

    mybestsqs = [] #guardar tablas q de todos los iniciales
    for nodoinicial in range(dimension):
        nodoiniciales = []
        for carinit in range(cars_number):
            CompQ_table = Q_table.copy()
            mask = np.array([True] * dimension)
            maskcars = np.array([True] * cars_number)
            route = []
            cars = []
            cache_distance_best = np.zeros((EPOCHS,))
            cache_distance_comp = np.zeros((EPOCHS,))
            
            
            for ep in range(EPOCHS):
                CompQ_table = eps_greedy_update(nodoinicial, carinit, CompQ_table, mask, maskcars, route, cars)
                greedy_cost = compute_value_of_q_table(Q_table, nodoinicial)
                greedy_cost_comp = compute_value_of_q_table(CompQ_table, nodoinicial)
                #cache_distance_best[ep] = greedy_cost
                #cache_distance_comp[ep] = greedy_cost_comp

                if greedy_cost_comp[0]+greedy_cost_comp[1] < greedy_cost[0]+greedy_cost[1]:
                    Q_table[:, :, :] = CompQ_table[:, :, :]
                # resetting route and mask for next episode
                route = []
                cars = []
                mask[:] = True
                maskcars[:] = True
                
            greedy_cost, greedy_route, greedy_cars = lastcomput(Q_table, nodoinicial)

            nodoiniciales.append([greedy_cost, greedy_route, greedy_cars])
        mybestsqs.append(nodoiniciales)

    mejor_costo = [float('inf'),float('inf')]
    mejor_comb = None  # Para guardar (nodoinicial, carinit)
    mejor_ruta = None
    mejor_autos = None

    for nodoinicial in range(dimension):
        for carinit in range(cars_number):
            costo = (mybestsqs[nodoinicial][carinit])[0]
            ruta = (mybestsqs[nodoinicial][carinit])[1]
            autos = (mybestsqs[nodoinicial][carinit])[2]

            if costo[0]+costo[1] < mejor_costo[0] + mejor_costo[1]:
                mejor_costo = costo
                mejor_comb = (nodoinicial, carinit)
                mejor_ruta = ruta
                mejor_autos = autos

    return mejor_costo, mejor_ruta, mejor_autos

import time

def evaluar_multiples(Q_table, runs=10):
    costos = []
    tiempos = []
    mejor = {
        'cost': float('inf'),
        'time': float('inf'),
        'ruta': None,
        'autos': None
    }

    for _ in range(runs):
        start = time.time()
        costo, ruta, autos = solving_qlearning(Q_table.copy())
        t = time.time() - start

        total = costo[0] + costo[1]
        costos.append(total)
        tiempos.append(t)

        if total < mejor['cost']:
            mejor.update({'cost': total, 'ruta': ruta, 'autos': autos})
        if t < mejor['time']:
            mejor['time'] = t

    promedio_costo = np.mean(costos)
    promedio_tiempo = np.mean(tiempos)

    return {
        'promedio_costo': promedio_costo,
        'promedio_tiempo': promedio_tiempo,
        'mejor_costo': mejor['cost'],
        'mejor_tiempo': mejor['time'],
        'mejor_ruta': mejor['ruta'],
        'mejores_autos': mejor['autos']
    }

# Uso:
resultados = evaluar_multiples(Q_table, runs=10)

with open("resultadosq.txt", "a", encoding="utf-8") as f:
    print(file=f)  # línea vacía de separación
    print(name + " Qlearning", file=f)
    print(f"Costo medio: {resultados['promedio_costo']:.2f}", file=f)
    print(f"Tiempo medio: {resultados['promedio_tiempo']:.4f} s", file=f)
    print(f"Mejor costo: {resultados['mejor_costo']:.2f}", file=f)
    print(f"Mejor tiempo: {resultados['mejor_tiempo']:.4f} s", file=f)
    print("Mejor ruta:", resultados['mejor_ruta'], file=f)
    print("Mejores autos:", resultados['mejores_autos'], file=f)
