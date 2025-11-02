import random
import os

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = []
        self.adj = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.edges.append((u, v))
        self.adj[u].add(v)
        self.adj[v].add(u)

    def degree(self, vertex):
        return len(self.adj[vertex])

    def neighbors(self, vertex):
        return self.adj[vertex]


class AntColonyGraphColoring:
    def __init__(self, graph, n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, rho=0.1, q=1.0):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        self.pheromone = [[1.0] * graph.num_vertices for _ in range(graph.num_vertices)]
        self.best_coloring = {}
        self.best_color_count = float('inf')

    def solve(self):
        for i in range(self.n_iterations):
            iteration_best_coloring = {}
            iteration_best_color_count = float('inf')

            for _ in range(self.n_ants):
                coloring, num_colors = self._construct_solution_for_ant()
                if num_colors < iteration_best_color_count:
                    iteration_best_color_count = num_colors
                    iteration_best_coloring = coloring

            if iteration_best_color_count < self.best_color_count:
                self.best_color_count = iteration_best_color_count
                self.best_coloring = iteration_best_coloring

            self._update_pheromones()

            if (i + 1) % 10 == 0:
                print(f"Ітерація {i + 1}: Найкраща кількість кольорів = {self.best_color_count}")

        return self.best_coloring, self.best_color_count

    def _construct_solution_for_ant(self):
        coloring = {}
        uncolored_vertices = list(range(self.graph.num_vertices))
        random.shuffle(uncolored_vertices)

        while uncolored_vertices:
            vertex = self._select_next_vertex(uncolored_vertices, coloring)
            color = self._select_color_for_vertex(vertex, coloring)
            coloring[vertex] = color
            uncolored_vertices.remove(vertex)

        num_colors = len(set(coloring.values()))
        return coloring, num_colors

    def _select_next_vertex(self, uncolored_vertices, coloring):
        max_saturation_degree = -1
        next_vertex = -1

        for vertex in uncolored_vertices:
            neighbor_colors = {coloring[n] for n in self.graph.neighbors(vertex) if n in coloring}
            saturation_degree = len(neighbor_colors)

            if saturation_degree > max_saturation_degree:
                max_saturation_degree = saturation_degree
                next_vertex = vertex
            elif saturation_degree == max_saturation_degree:
                if self.graph.degree(vertex) > self.graph.degree(next_vertex):
                    next_vertex = vertex

        return next_vertex if next_vertex != -1 else uncolored_vertices[0]

    def _select_color_for_vertex(self, vertex, coloring):
        neighbor_colors = {coloring[n] for n in self.graph.neighbors(vertex) if n in coloring}

        color_probabilities = {}
        total_prob = 0.0
        used_colors = set(coloring.values())

        for color in range(self.graph.num_vertices):
            if color not in neighbor_colors:
                pheromone = self.pheromone[vertex][color] ** self.alpha

                heuristic = (1.0 if color in used_colors else 0.5) ** self.beta

                prob = pheromone * heuristic
                color_probabilities[color] = prob
                total_prob += prob

        if total_prob == 0:
            return max(used_colors) + 1 if used_colors else 0

        rand_val = random.uniform(0, total_prob)
        cumulative_prob = 0
        for color, prob in color_probabilities.items():
            cumulative_prob += prob
            if cumulative_prob >= rand_val:
                return color

        return list(color_probabilities.keys())[-1]  # Fallback

    def _update_pheromones(self):
        for i in range(self.graph.num_vertices):
            for j in range(self.graph.num_vertices):
                self.pheromone[i][j] *= (1 - self.rho)

        if self.best_color_count != float('inf'):
            delta_tau = self.q / self.best_color_count
            for vertex, color in self.best_coloring.items():
                self.pheromone[vertex][color] += delta_tau


def parse_dimacs_col_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_vertices, num_edges = 0, 0
    for line in lines:
        if line.startswith('p edge'):
            _, _, nv_str, ne_str = line.split()
            num_vertices = int(nv_str)
            num_edges = int(ne_str)
            break

    if num_vertices == 0:
        raise ValueError("Не знайдено 'p edge' у файлі.")

    graph = Graph(num_vertices)
    for line in lines:
        if line.startswith('e'):
            _, u_str, v_str = line.split()
            u, v = int(u_str) - 1, int(v_str) - 1
            graph.add_edge(u, v)

    return graph


def run_test(filename):
    print(f"\n\n{'=' * 25} ЗАПУСК ТЕСТУ: {filename} {'=' * 25}")

    try:
        graph = parse_dimacs_col_file(filename)
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{filename}' не знайдено.")
        return
    except Exception as e:
        print(f"ПОМИЛКА при читанні файлу '{filename}': {e}")
        return

    print(f"Граф завантажено: {graph.num_vertices} вершин, {len(graph.edges)} ребер.")

    if graph.num_vertices < 30:
        iterations, ants, beta = 100, 10, 2.0
    elif graph.num_vertices < 80:
        iterations, ants, beta = 150, 15, 3.0
    else:
        iterations, ants, beta = 200, 20, 5.0

    aco_solver = AntColonyGraphColoring(
        graph=graph,
        n_ants=ants,
        n_iterations=iterations,
        alpha=1.0,
        beta=beta,
        rho=0.1,
        q=1.0
    )

    best_coloring, best_color_count = aco_solver.solve()

    print("\n" + "-" * 40)
    print(f"РЕЗУЛЬТАТИ ДЛЯ '{filename}'")
    if best_coloring:
        print(f"Знайдено розфарбування з {best_color_count} кольорами.")

        print("Приклад розфарбування (перші 15 вершин):")
        for i in sorted(best_coloring.keys())[:15]:
            print(f"  Вершина {i + 1}: Колір {best_coloring[i]}")
    else:
        print("Не вдалося знайти розв'язок.")
    print("-" * 40)


if __name__ == '__main__':
    test_files = ["myciel3.col", "queen5_5.col", "jean.col"]

    for file in test_files:
        run_test(file)