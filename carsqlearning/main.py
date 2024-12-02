import time
import numpy as np
import networkx as nx
from numba import njit
from src.model import QLearning
from src.utils import (
    compute_greedy_route,
    route_distance,
    write_overall_results
)

from parser_txt_file import parse_txt_file

# Constantes
EPOCHS = 4000
LEARNING_RATE = 0.2
GAMMA = 0.95
EPSILON = 0.1
SEED = 42

def solve(file_path, start_node):
    # Cargar datos desde el archivo
    dimension, edge_weight_matrices, return_rate_matrices = parse_txt_file(file_path)

    # Número de ciudades y vehículos
    n_cities = dimension
    n_vehicles = len(edge_weight_matrices)

    results = {}

    for kk in range(10):  # Realizar varias ejecuciones para cada configuración
        start_time = time.time()

        # Crear grafos para cada vehículo
        graphs = []
        for v in range(n_vehicles):
            graph = nx.DiGraph()
            for i in range(n_cities):
                for j in range(n_cities):
                    if edge_weight_matrices[v][i][j] > 0:
                        graph.add_edge(i, j, weight=edge_weight_matrices[v][i][j])
            graphs.append(graph)

        # Inicializar matriz Q para cada vehículo
        Q_tables = [np.zeros((n_cities, n_cities)) for _ in range(n_vehicles)]

        # Entrenar Q-learning para cada vehículo
        best_routes = []
        total_costs = []

        for v in range(n_vehicles):
            Q_tables[v], _, _ = QLearning(
                Q_tables[v], edge_weight_matrices[v], epsilon=EPSILON, gamma=GAMMA,
                lr=LEARNING_RATE, epochs=EPOCHS, start_node=start_node
            )

            # Obtener la mejor ruta comenzando desde el nodo inicial
            greedy_route = compute_greedy_route(Q_tables[v], start_node).tolist()
            greedy_route.append(greedy_route[0])  # Completar el ciclo
            route_cost = route_distance(np.array(greedy_route), edge_weight_matrices[v])
            best_routes.append(greedy_route)
            total_costs.append(route_cost)

        # Calcular el costo total combinando rutas y retornos
        combined_cost = sum(total_costs)
        for v in range(n_vehicles):
            for i in range(len(best_routes[v]) - 1):
                origin = best_routes[v][i]
                destination = best_routes[v][i + 1]
                combined_cost += return_rate_matrices[v][origin][destination]

        elapsed_time = time.time() - start_time
        results[f"Run_{kk}"] = {"Total Cost": combined_cost, "Time": elapsed_time}

        print(f"Iteración {kk+1}: Costo Total = {combined_cost}, Tiempo = {elapsed_time:.2f} segundos")

    return results

if __name__ == "__main__":
    file_path = "temp copy.txt"
    start_node = 0  # Nodo desde el cual iniciar
    results = solve(file_path, start_node)
    write_overall_results(results, None, "_combined_optimized")
