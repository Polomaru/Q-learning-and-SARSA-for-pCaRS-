import numpy as np
from numba import njit

@njit
def sarsa_update(Q_table, edge_weight_matrices, return_rate_matrices, mask, route, epsilon, gamma, lr, N):

    def calcular_costo_total(nodo_origen, nodo_destino, carro_idx):
        edge_cost = edge_weight_matrices[carro_idx][nodo_origen][nodo_destino]
        return edge_cost
    
    
    """ Actualiza la tabla Q usando SARSA """
    mask[0] = False
    next_visit = 0
    current = 0
    reward = 0

    # Acción inicial usando epsilon-greedy
    possible = np.where(mask == True)[0]
    if len(possible) == 1:
        next_visit = possible[0]
    else:
        u = np.random.random()
        if u < epsilon:
            next_visit = np.random.choice(possible)
        else:
            next_visit, _ = custom_argmax(Q_table, int(current), mask)

    route[0] = current
    route[1] = next_visit

    for i in range(1, N):
        next_visit = int(next_visit)

        # Recompensa y actualización de la máscara
        reward = -calcular_costo_total(current, next_visit, route[i-1])
        mask[next_visit] = False

        # Elegir siguiente acción con epsilon-greedy
        possible = np.where(mask == True)[0]
        if len(possible) == 0:
            next_next_visit = route[0]  # Volver al inicio
            next_q_value = Q_table[next_visit, next_next_visit]
        else:
            if np.random.random() < epsilon:
                next_next_visit = np.random.choice(possible)
            else:
                next_next_visit, next_q_value = custom_argmax(Q_table, next_visit, mask)

        # Actualización de SARSA
        Q_table[current, next_visit] += lr * (reward + gamma * Q_table[next_visit, next_next_visit] - Q_table[current, next_visit])

        # Actualizar estado y acción
        current = next_visit
        next_visit = next_next_visit
        if i < N - 1:
            route[i + 1] = next_visit

    return Q_table

def sarsa(Q_table, edge_weight_matrices, return_rate_matrices, epsilon, gamma, lr, epochs):
    """ Ejecuta el algoritmo SARSA """
    N = Q_table.shape[0]
    mask = np.array([True] * N)
    route = np.zeros((N,))
    for ep in range(epochs):
        Q_table = sarsa_update(Q_table, edge_weight_matrices, return_rate_matrices, mask, route, epsilon, gamma, lr, N)
        route[:] = 0
        mask[:] = True
    return Q_table

def custom_argmax(Q_table, current, mask):
    """ Encuentra la acción con el máximo valor Q """
    valid_actions = np.where(mask == True)[0]
    q_values = Q_table[current, valid_actions]
    max_action_idx = valid_actions[np.argmax(q_values)]
    return max_action_idx, q_values[np.argmax(q_values)]