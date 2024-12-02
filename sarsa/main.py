import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit

from src.model import SARSA
from src.utils import (
    compute_greedy_route,
    load_data,
    route_distance,
    trace_progress,
    write_overall_results
)

# Constantes
EPOCHS = 4000
LEARNING_RATE = 0.2
GAMMA = 0.95
EPSILON = 0.1
SEED = 42
NODE_COLOR = "skyblue"
ROUTE_NODE_COLOR = "orange"
ROUTE_EDGE_COLOR = "red"

def draw_graph_with_route(graph, route, positions, title):
    """
    Dibuja el grafo resaltando la ruta óptima.
    
    Args:
        graph (nx.DiGraph): Grafo dirigido con pesos.
        route (list): Secuencia de nodos que define la ruta.
        positions (dict): Posiciones de los nodos para el layout.
        title (str): Título del gráfico.
    """
    # Construir aristas de la ruta
    route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]

    # Atributos de aristas
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    plt.figure(figsize=(10, 8))
    # Dibujar nodos y aristas generales
    nx.draw(
        graph, positions, with_labels=True, node_color=NODE_COLOR,
        node_size=800, font_size=10, font_weight="bold"
    )
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_color="black")

    # Resaltar la ruta
    nx.draw_networkx_edges(
        graph, positions, edgelist=route_edges, edge_color=ROUTE_EDGE_COLOR,
        width=2, arrows=True, arrowstyle='-|>', arrowsize=20
    )
    nx.draw_networkx_nodes(graph, positions, nodelist=route, node_color=ROUTE_NODE_COLOR, node_size=900)

    # Mostrar la ruta en el lado derecho
    route_text = "\n".join([f"{node} ->" for node in route[:-1]]) + f"{route[-1]}"
    plt.gcf().text(
        0.85, 0.5, route_text, fontsize=12, ha='left', va='center',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.title(title)
    plt.subplots_adjust(right=0.8)
    plt.show()

data = load_data()
mydicto = {}

def solve_tsp():
    global data, mydicto
    """
    Resuelve instancias TSP utilizando Q-learning y genera visualizaciones.
    """
    
    
    for cities, (distance_matrix, optimal_cost) in data.items():
        for kk in range(10):
            #print(f"Resolviendo TSP para {cities} ciudades...")
            start_time = time.time()
            # Crear grafo
            graph = nx.DiGraph()
            for i in range(cities):
                for j in range(cities):
                    if distance_matrix[i][j] > 0:
                        graph.add_edge(i, j, weight=distance_matrix[i][j])

            positions = nx.spring_layout(graph, seed=SEED)
            Q_table = np.zeros((cities, cities))

            # Entrenar Q-learning
            
            Q_table, best_distances, comp_distances = SARSA(
                Q_table, distance_matrix, epsilon=EPSILON, gamma=GAMMA,
                lr=LEARNING_RATE, epochs=EPOCHS
            )

            #print(f"\nMatriz Q-Learning para {cities} ciudades:")
            #print(Q_table)
            # Visualizar progreso
            #trace_progress(comp_distances, optimal_cost, f"{cities}_Cities_Exploration")
            #trace_progress(best_distances, optimal_cost, f"{cities}_Cities_Best_Solution")

            # Calcular ruta óptima
            
            greedy_route = compute_greedy_route(Q_table).tolist()
            greedy_route.append(greedy_route[0])  # Completar el ciclo
            #draw_graph_with_route(graph, greedy_route, positions, f"TSP - {cities} ciudades")

            # Guardar resultados
            
            route_cost = route_distance(np.array(greedy_route), distance_matrix)
            
            
            elapsed_time = time.time() - start_time
            mydicto[str(cities) + "_" + str(kk)] = [route_cost, elapsed_time]
            print(f"Tiempo total: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    solve_tsp()
    write_overall_results(mydicto, data, "_optimized")
