import numpy as np
import networkx as nx
import pandas as pd
import csv
import ast
import math
import heapq
import time
from collections import Counter
import cvxpy as cvx
from multiprocessing import cpu_count
import copy





_nbr_topk = 3000
_apsp = {}


def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
        x_clipped = np.clip(x, -709, 709)
        a, b = 0, 1  # Define the target range
        return a + (b - a) / (1 + np.exp(-x_clipped))




def delete_top_weighted_edges_direct(G, percentage=0.08):
    all_edges = list(G.edges(data=True))
    all_edges_sorted = sorted(all_edges, key=lambda x: adjusted_sigmoid_0_to_1(x[2].get('weight', 1)), reverse=True)
    n = int(len(all_edges_sorted) * percentage)
    top_edges = [(u, v) for u, v, d in all_edges_sorted[:n]]
    G.remove_edges_from(top_edges)
    # Remove isolated nodes (recommended)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Removed {n} edges with the highest weights.")

def extract_core_metrics_each_component(G_original, G_after_surgery, weight="weight", remove_ratio=0.08):
    """
    In the post-surgery graph:
    - Remove high-weight edges
    - Traverse all connected components and induce subgraphs from the original graph
    - Compute and print core metrics for each component
    - Output the top two components with the highest r_d
    """
    # Copy the graph to avoid modifying the original
    G_surgery = G_after_surgery.copy()

    # Remove high-weight edges
    all_edges = list(G_surgery.edges(data=True))
    sorted_edges = sorted(all_edges, key=lambda x: x[2].get(weight, 0), reverse=True)
    num_edges_to_remove = int(len(sorted_edges) * remove_ratio)
    edges_to_remove = [(u, v) for (u, v, d) in sorted_edges[:num_edges_to_remove]]
    G_surgery.remove_edges_from(edges_to_remove)

    # Get all connected components
    components = list(nx.connected_components(G_surgery))
    print(f"Number of remaining connected components after surgery: {len(components)}")

    # Store metrics for each component
    metrics_list = []

    # Iterate through each connected component, compute metrics, and print
    for idx, comp_nodes in enumerate(components):
        core_candidate = G_original.subgraph(comp_nodes).copy()
        print(f"\n>>> Connected Component {idx + 1}:")
        print(f"Number of nodes: {core_candidate.number_of_nodes()}, edges: {core_candidate.number_of_edges()}")

        r_d, r_s = compute_core_graph_metrics(G_original, core_candidate)
        print(f"Core cohesion r_d = {r_d:.4f}, Residual graph average distance stretch r_s = {r_s:.4f}")

        metrics_list.append((idx, r_d, r_s, core_candidate))

    # Output the top two components with highest r_d
    if len(metrics_list) >= 2:
        print("\n>>> Top 2 connected components by r_d:")
        top2 = sorted(metrics_list, key=lambda x: x[1], reverse=True)[:2]
        for rank, (idx, r_d, r_s, core_candidate) in enumerate(top2, 1):
            print(f"\n### Top {rank}: Connected Component {idx + 1}")
            print(f"Number of nodes: {core_candidate.number_of_nodes()}, edges: {core_candidate.number_of_edges()}")
            print(f"r_d = {r_d:.4f}, r_s = {r_s:.4f}")


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

    def ricci_flow_iteration(self, iterations=50, delta=1e-6):
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue Ricci flow.")
        else:
            self.compute_ricci_curvature()
            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        for i in range(iterations):
            # Ricci flow iteration
            for (v1, v2) in self.G.edges():
                rc = self.G[v1][v2]["ricciCurvature"]
                w_old = self.G[v1][v2][self.weight]
                w_new = adjusted_sigmoid_0_to_1(w_old * (1 - rc))
                self.G[v1][v2][self.weight] = w_new

            self.compute_ricci_curvature()
            print(f"=== Ricci flow iteration {i} ===")

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())
            print(f"Ricci curvature difference: {diff:.4f}")

            if diff < delta:
                print("Ricci curvature converged, stopping.")
                break

            self.densities = {}

        print(f"\n{time.time() - t0:.6f} secs for Ricci flow computation.")

def run_ricci_flow_surgery(G_original, alpha=0.1, iterations=40, surgery_every=2, delete_ratio=0.08):
    G = G_original.copy()
    ricci = RhoCurvature(G, alpha=alpha)
    ricci.compute_ricci_curvature()

    for i in range(iterations):
        print(f"\n=== Ricci Flow Iteration {i+1} ===")
        ricci.ricci_flow_iteration(iterations=1)

        if (i + 1) % surgery_every == 0:
            delete_top_weighted_edges_direct(ricci.G, percentage=delete_ratio)
            print(f"After surgery iteration {(i+1)//surgery_every}: Nodes={ricci.G.number_of_nodes()}, Edges={ricci.G.number_of_edges()}")

    # Print core metrics of all connected components after surgery, do not return core subgraph
    extract_core_metrics_each_component(G_original=G, G_after_surgery=ricci.G)


def build_residual_subgraph_from_core_nodes(original_G: nx.Graph, core_nodes: set):
    residual_nodes = set(original_G.nodes) - core_nodes
    residual_subgraph = original_G.subgraph(residual_nodes).copy()
    print(f"Building residual subgraph — Nodes:{residual_subgraph.number_of_nodes()}，Edges：{residual_subgraph.number_of_edges()}")
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


# 1. Load the original graph
#G = nx.read_gexf(r"D:\PythonRiccicurvature\karate.gexf")
G= nx.read_gexf(r"C:\Users\22118\Downloads\citeseer.gexf")
#G =nx.read_gexf(r"C:\Users\22118\Downloads\cora.gexf")
#edge_file = r"C:\Users\22118\AppData\Local\Temp\0c9fa094-c7e4-4d94-8978-f81238834841_bio-CE-HT.zip.841\bio-CE-HT.edges"  # 替换为你的文件路径
#df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target', 'weight'])
#G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight')

print(f"Original graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")



# Run Ricci flow + surgery + print metrics for all connected components
run_ricci_flow_surgery(
    G,
    alpha=0.1,
    iterations=40,
    surgery_every=2,
    delete_ratio=0.08
)



