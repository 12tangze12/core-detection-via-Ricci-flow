import numpy as np
import math
import time
from multiprocessing import cpu_count
import cvxpy as cvx
import copy
import heapq
import networkx as nx
import pandas as pd


_nbr_topk = 3000
_apsp = {}


class RhoCurvature:

    def __init__(self, G, alpha=0.1, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e

    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        self.lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return self.lengths

    def _get_single_node_neighbors_distributions(self, node):
        neighbors = list(self.G.neighbors(node))
        heap_weight_node_pair = []
        for nbr in neighbors:
            w = self.G[node][nbr][self.weight]
            if len(heap_weight_node_pair) < _nbr_topk:
                heapq.heappush(heap_weight_node_pair, (w, nbr))
            else:
                heapq.heappushpop(heap_weight_node_pair, (w, nbr))
        nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])
        if not neighbors:
            return [1], [node]
        if nbr_edge_weight_sum > self.EPSILON:
            distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]

        else:
            # Sum too small, just evenly distribute to every neighbors

            distributions = [(1.0 - self.alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)
        nbr = [x[1] for x in heap_weight_node_pair]
        return distributions + [self.alpha], nbr + [node]

    def _get_edge_density_distributions(self):
        densities = dict()

        for x in self.G.nodes():
            densities[x] = self._get_single_node_neighbors_distributions(x)

    def _optimal_transportation_distance(self, x, y, d):
        rho = cvx.Variable((len(y), len(x)))

        # objective function d(x,y) * rho * x, need to do element-wise multiply here
        obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

        # \sigma_i rho_{ij}=[1,1,...,1]
        source_sum = cvx.sum(rho, axis=0, keepdims=True)
        constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        return m

    def _distribute_densities(self, source, target):
        # Append source and target node into weight distribution matrix x,y
        x, source_topknbr = self._get_single_node_neighbors_distributions(source)
        y, target_topknbr = self._get_single_node_neighbors_distributions(target)
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(self.lengths[src][tgt])
            d.append(tmp)
        d = np.array(d)
        x = np.array(x)
        y = np.array(y)
        return x, y, d

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = 1 - m / self.lengths[source][target]  # Divided by the length of d(i, j)
        # print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    # def _wrap_compute_single_edge(self, stuff):
    #     return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):

        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        # if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()

        # Start compute edge Ricci curvature

        #     p = Pool(processes=self.proc)

        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = [self._compute_ricci_curvature_single_edge(*arg) for arg in args]

        #    result = p.map_async(self._wrap_compute_single_edge, args).get()
        #   p.close()
        #  p.join()

        return result

    def compute_ricci_curvature(self):

        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())

        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

    def ricci_flow_iteration(self, iterations=50, step=0.1, delta=1e-6):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(self.G)

        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()
            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        self.rc_diff = []
        for i in range(iterations):
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * (self.lengths[v1][v2])

            w = nx.get_edge_attributes(self.G, self.weight)

            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1, v2 in list(G1.edges()):
                    if G1[v1][v2][self.weight] < delta * 10:
                        # Record the adjacent edge weights of the two nodes
                        neighbors_v1 = {nbr: G1[v1][nbr][self.weight] for nbr in G1.neighbors(v1)}
                        neighbors_v2 = {nbr: G1[v2][nbr][self.weight] for nbr in G1.neighbors(v2)}

                        # Contract the edge
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)

                        # Get the new node name after contraction (defaults to v1)
                        new_node = v1

                        # For each neighbor of the new node, update edge weight to the minimum of the original edges
                        for nbr in G1.neighbors(new_node):
                            w1 = neighbors_v1.get(nbr, float('inf'))
                            w2 = neighbors_v2.get(nbr, float('inf'))
                            min_w = min(w1, w2)
                            G1[new_node][nbr][self.weight] = min_w

                        merged = True
                        print(f"Contracted edge: ({v1}, {v2}) with updated weights")
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print(f"=== Ricci flow iteration {i} ===")

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print(f"Ricci curvature difference: {diff}")
            print(f"max:{max(rc.values())}, min:{min(rc.values())} | maxw:{max(w.values())}, minw:{min(w.values())}")

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            self.densities = {}  # reset cache

        print(f"\n{time.time() - t0:.6f} secs for Ricci flow computation.")




def extract_core_subgraph(G, weight="weight", remove_ratio=0.8):
    G_copy = G.copy()
    original_G = G.copy()

    total_nodes = original_G.number_of_nodes()
    target_core_size = total_nodes // 2  # Target: keep about half the nodes
    print(f"Target number of core nodes: {target_core_size}")

    # Step 1: Sort and remove edges by weight
    all_edges = list(G_copy.edges(data=True))
    sorted_edges = sorted(all_edges, key=lambda x: x[2].get(weight, 0), reverse=True)
    num_edges_to_remove = int(len(sorted_edges) * remove_ratio)
    edges_to_remove = [(u, v) for (u, v, d) in sorted_edges[:num_edges_to_remove]]
    G_copy.remove_edges_from(edges_to_remove)

    print(f"[Pruning Phase] Number of edges removed: {num_edges_to_remove}")
    print(f"[Pruning Phase] Remaining edges: {G_copy.number_of_edges()}")

    # Step 2: Identify non-isolated and isolated nodes
    isolated_nodes = list(nx.isolates(G_copy))
    non_isolated_nodes = [node for node in G_copy.nodes if node not in isolated_nodes]
    print(f"Non-isolated nodes: {len(non_isolated_nodes)}")
    print(f"Isolated nodes: {len(isolated_nodes)}")

    # Step 3: Optionally add back top-degree isolated nodes to reach target_core_size
    isolated_degrees = {node: original_G.degree(node) for node in isolated_nodes}
    sorted_isolated = sorted(isolated_degrees.items(), key=lambda x: x[1], reverse=True)

    num_to_add = target_core_size - len(non_isolated_nodes)
    top_isolated_nodes = [node for node, deg in sorted_isolated[:max(num_to_add, 0)]]


    core_nodes = set(non_isolated_nodes) | set(top_isolated_nodes)
    print(f"Final number of core nodes: {len(core_nodes)}")

    # Step 4: Create subgraph with selected core nodes
    core_subgraph = original_G.subgraph(core_nodes).copy()
    print(f"Core subgraph - Nodes: {core_subgraph.number_of_nodes()}, Edges: {core_subgraph.number_of_edges()}")

    return core_subgraph


def build_residual_subgraph_from_core_nodes(original_G: nx.Graph, core_nodes: set):
    residual_nodes = set(original_G.nodes) - core_nodes
    residual_subgraph = original_G.subgraph(residual_nodes).copy()
    print(f"Building residual subgraph — Nodes: {residual_subgraph.number_of_nodes()}, Edges: {residual_subgraph.number_of_edges()}")
    return residual_subgraph




def compute_core_graph_metrics(G: nx.Graph, core_G: nx.Graph):
    core_nodes = set(core_G.nodes)
    residual_nodes = set(G.nodes) - core_nodes

    print("\n===== [1] Unweighted Core Cohesion r_d Calculation =====")
    numerator = sum(
        len(core_G.edges(n)) / len(G.edges(n))
        for n in core_nodes if len(G.edges(n)) > 0
    )
    core_node_count = len(core_nodes)
    r_d = numerator / core_node_count if core_node_count > 0 else 0.0
    print(f"[DEBUG] numerator = {numerator:.4f}, core_node_count = {core_node_count}, r_d = {r_d:.4f}")

    print("\n===== [2] Average Distance Stretch r_s =====")
    G_res = build_residual_subgraph_from_core_nodes(G, core_nodes)
    print(f"[DEBUG] Number of residual nodes: {len(residual_nodes)}")
    print(f"[DEBUG] Residual subgraph nodes: {G_res.number_of_nodes()}")
    print(f"[DEBUG] Residual subgraph edges: {G_res.number_of_edges()}")

    path_length_G = dict(nx.all_pairs_shortest_path_length(G))
    path_length_Gres = dict(nx.all_pairs_shortest_path_length(G_res))

    stretch_sum = 0.0
    connected_pair_count = 0
    residual_node_list = list(residual_nodes)

    for i in range(len(residual_node_list)):
        for j in range(i + 1, len(residual_node_list)):
            u, v = residual_node_list[i], residual_node_list[j]
            try:
                dist_G = path_length_G[u][v]
                dist_Gres = path_length_Gres[u][v]
                stretch_sum += dist_Gres / dist_G
                connected_pair_count += 1
            except KeyError:
                continue  # Skip if nodes are disconnected in residual graph

    r_s = stretch_sum / connected_pair_count if connected_pair_count > 0 else float('inf')
    print(f"[DEBUG] Successfully paired nodes: {connected_pair_count}, r_dist_stretch = {r_s:.4f}")

    return r_d, r_s


def print_core_metrics(r_d, r_s):
    print(f"Core weighted cohesion r_d = {r_d:.4f}")
    print(f"Average distance stretch of residual graph r_s = {r_s:.4f}")


# ===== Extract Core Nodes =====
def extract_core_nodes(G: nx.Graph, method: str, core_size: int) -> set:
    """
    Extract a set of core nodes based on the specified method,
    and return the largest connected component as the final core subgraph nodes.

    Supported methods:
      - 'degree'
      - 'betweenness'
      - 'closeness'
      - 'pagerank'
    """
    if method == 'degree':
        centrality = nx.degree_centrality(G)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif method == 'pagerank':
        centrality = nx.pagerank(G, alpha=0.85, weight=None)
    else:
        raise ValueError(
            "Unsupported method. Choose from: 'degree', 'betweenness', 'closeness', 'pagerank'.")

    # Select top-k nodes by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:core_size]]

    # Construct core subgraph
    core_subgraph = G.subgraph(top_nodes)

    # Extract the largest connected component
    largest_cc = max(nx.connected_components(core_subgraph), key=len)
    return set(largest_cc)


def main():
    # === Load the graph ===
    # Citeseer dataset
    G = nx.read_gexf(r"C:\Users\22118\Downloads\citeseer.gexf")
    # Cora dataset
    # G = nx.read_gexf(r"C:\Users\22118\Downloads\cora.gexf")
    # Bio dataset
    # edge_file = r"C:\Users\22118\AppData\Local\Temp\0c9fa094-c7e4-4d94-8978-f81238834841_bio-CE-HT.zip.841\bio-CE-HT.edges"
    # df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target', 'weight'])
    # G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight')

    print(f"Original graph - nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    G_original = copy.deepcopy(G)

    # === Ricci Flow Core Extraction ===
    rho_curv = RhoCurvature(copy.deepcopy(G), alpha=0.1, weight="weight", proc=cpu_count())
    print("Computing initial Ricci curvature...")
    rho_curv.compute_ricci_curvature()
    print("Starting Ricci flow evolution...")
    rho_curv.ricci_flow_iteration(iterations=12, step=0.1, delta=1e-6)

    print("Starting surgery operation...")
    core_subgraph_raw = extract_core_subgraph(copy.deepcopy(rho_curv.G), weight=rho_curv.weight, remove_ratio=0.8)
    largest_component = max(nx.connected_components(core_subgraph_raw), key=len)
    core_subgraph_largest = core_subgraph_raw.subgraph(largest_component).copy()
    core_nodes_largest = set(core_subgraph_largest.nodes())

    print("\n==== Core graph from Ricci flow (largest connected component) ====")
    print(f"Core nodes: {len(core_nodes_largest)}")
    print(f"Core edges: {core_subgraph_largest.number_of_edges()}")
    print_core_metrics(*compute_core_graph_metrics(G_original, core_subgraph_largest))

    # === Compare with other core extraction methods ===
    core_sizes = {
        'pagerank': 484,
        'degree': 416,
        'betweenness': 345,
        'closeness': 343,
    }

    core_nodes_all = {'Ricci Flow': core_nodes_largest}

    methods = ['pagerank', 'degree', 'betweenness', 'closeness']
    for method in methods:
        core_size = core_sizes.get(method, len(core_nodes_largest))
        print(f"\n==== Extracting core subgraph using {method} (core size: {core_size}) ====")
        core_nodes = extract_core_nodes(G_original, method, core_size)

        # Extract the largest connected component
        core_subgraph = G_original.subgraph(core_nodes)
        largest_cc = max(nx.connected_components(core_subgraph), key=len)
        core_nodes_largest_method = set(largest_cc)
        core_subgraph_largest_method = core_subgraph.subgraph(largest_cc).copy()

        core_nodes_all[method.capitalize()] = core_nodes_largest_method

        print(f"Core nodes: {len(core_nodes_largest_method)}, core edges: {core_subgraph_largest_method.number_of_edges()}")
        print_core_metrics(*compute_core_graph_metrics(G_original, core_subgraph_largest_method))




if __name__ == "__main__":
    main()


