import numpy as np
from numba import njit
from src.utils import compute_value_of_q_table, custom_argmax


@njit
def eps_greedy_update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    N: int,
):
    mask[0] = False
    next_visit = 0
    reward = 0
    for i in range(1, N):
        # Iteration i : choosing ith city to visit
        possible = np.where(mask == True)[0]
        current = route[i - 1]
        if len(possible) == 1:
            next_visit = possible[0]
            reward = -distances[int(current), int(next_visit)]
            # Reward for finishing the route
            max_next = -distances[int(next_visit), int(route[0])]
        else:
            u = np.random.random()
            if u < epsilon:
                # random choice amongst possible
                next_visit = np.random.choice(possible)
            else:
                next_visit, _ = custom_argmax(Q_table, int(current), mask)
            # update mask and route
            mask[next_visit] = False
            route[i] = next_visit
            reward = -distances[int(current), int(next_visit)]
            # Get max Q from new state
            _, max_next = custom_argmax(Q_table, int(next_visit), mask)
        # updating Q
        Q_table[int(current), int(next_visit)] = Q_table[
            int(current), int(next_visit)
        ] + lr * (reward + gamma * max_next - Q_table[int(current), int(next_visit)])
    return Q_table




@njit
def QLearning(
    Q_table: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    epochs: int,
    start_node: int,  # Nuevo parámetro: nodo de inicio
):
    """Realiza el algoritmo simple de Q-learning, epsilon-greedy, para resolver el TSP.

    Args:
        Q_table (np.ndarray): Tabla Q inicial.
        distances (np.ndarray): Matriz de distancias que describe la instancia del TSP.
        epsilon (float): Parámetro de exploración para epsilon-greedy.
        gamma (float): Peso para la recompensa futura.
        lr (float): Tasa de aprendizaje para la actualización de Q.
        epochs (int, opcional): Número de iteraciones. Por defecto es 100.
        start_node (int, opcional): Nodo de inicio. Por defecto es 0.

    Returns:
        np.ndarray: Tabla Q obtenida después del entrenamiento.
        list: Contiene las distancias greedy para cada época.
        list: Contiene las distancias comparativas para cada época.
    """
    N = Q_table.shape[0]
    CompQ_table = Q_table.copy()
    mask = np.array([True] * N)
    route = np.zeros((N,))
    cache_distance_best = np.zeros((epochs,))
    cache_distance_comp = np.zeros((epochs,))

    for ep in range(epochs):
        CompQ_table = eps_greedy_update(
            CompQ_table, distances, mask, route, epsilon, gamma, lr, N, start_node  # Pasamos start_node aquí
        )
        # Actualizar tabla Q solo si la mejor ruta encontrada hasta ahora mejora
        greedy_cost = compute_value_of_q_table(Q_table, distances)
        greedy_cost_comp = compute_value_of_q_table(CompQ_table, distances)
        cache_distance_best[ep] = greedy_cost
        cache_distance_comp[ep] = greedy_cost_comp

        if greedy_cost_comp < greedy_cost:
            Q_table[:, :] = CompQ_table[:, :]  # Copiar la tabla Q de la mejor solución

        # Reiniciar ruta y máscara para la siguiente época
        route[:] = 0
        mask[:] = True

    return Q_table, cache_distance_best, cache_distance_comp
