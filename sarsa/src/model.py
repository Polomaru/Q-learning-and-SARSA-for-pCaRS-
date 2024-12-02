import numpy as np
from numba import njit
from src.utils import compute_value_of_q_table, custom_argmax


@njit
def sarsa_update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    N: int,
):
    """Updates Q table using SARSA.

    Args:
        Q_table (np.ndarray): Input Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        mask (np.ndarray): Boolean mask giving which cities to ignore (already visited).
        route (np.ndarray): Route container.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.

    Returns:
        np.ndarray: Updated Q table.
    """
    mask[0] = False
    next_visit = 0
    current = 0
    reward = 0

    # Seleccionar la primera acción usando epsilon-greedy
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
        # Convertir índice a entero explícitamente
        next_visit = int(next_visit)

        # Actualizar recompensa y máscara
        reward = -distances[current, next_visit]
        mask[next_visit] = False

        # Elegir siguiente acción (next_next_visit) usando epsilon-greedy
        possible = np.where(mask == True)[0]
        if len(possible) == 0:
            # Si no hay más ciudades por visitar, volver al inicio
            next_next_visit = int(route[0])
            next_q_value = Q_table[next_visit, next_next_visit]
        else:
            if np.random.random() < epsilon:
                next_next_visit = np.random.choice(possible)
            else:
                next_next_visit, next_q_value = custom_argmax(
                    Q_table, next_visit, mask
                )

        # Actualización SARSA
        next_next_visit = int(next_next_visit)
        Q_table[current, next_visit] += lr * (
            reward + gamma * Q_table[next_visit, next_next_visit]
            - Q_table[current, next_visit]
        )

        # Avanzar al siguiente estado y acción
        current = next_visit
        next_visit = next_next_visit
        if i < N - 1:
            route[i + 1] = next_visit

    return Q_table



@njit
def SARSA(
    Q_table: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    epochs: int,
):
    """Performs SARSA algorithm to learn a solution to the TSP.

    Args:
        Q_table (np.ndarray): Initial Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.
        epochs (int, optional): Number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Q table obtained after training.
        list: contains greedy distances for each epoch.
    """
    N = Q_table.shape[0]
    CompQ_table = Q_table.copy()
    mask = np.array([True] * N)
    route = np.zeros((N,))
    cache_distance_best = np.zeros((epochs,))
    cache_distance_comp = np.zeros((epochs,))
    for ep in range(epochs):
        CompQ_table = sarsa_update(
            CompQ_table, distances, mask, route, epsilon, gamma, lr, N
        )
        # Update Q table only if the best found so far is improved
        greedy_cost = compute_value_of_q_table(Q_table, distances)
        greedy_cost_comp = compute_value_of_q_table(CompQ_table, distances)
        cache_distance_best[ep] = greedy_cost
        cache_distance_comp[ep] = greedy_cost_comp
        if greedy_cost_comp < greedy_cost:
            Q_table[:, :] = CompQ_table[:, :]
        # Resetting route and mask for the next episode
        route[:] = 0
        mask[:] = True
    return Q_table, cache_distance_best, cache_distance_comp
