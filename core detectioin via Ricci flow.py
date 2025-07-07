import numpy as np
import math
import time
from multiprocessing import cpu_count
import cvxpy as cvx
import copy
import heapq
from collections import Counter
import networkx as nx


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
                        # 记录两个节点的邻接边权重集合
                        neighbors_v1 = {nbr: G1[v1][nbr][self.weight] for nbr in G1.neighbors(v1)}
                        neighbors_v2 = {nbr: G1[v2][nbr][self.weight] for nbr in G1.neighbors(v2)}

                        # 合并边
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)

                        # 获取合并后新节点名（默认为v1）
                        new_node = v1

                        # 遍历新节点邻居，更新边权为合并前对应边权的最小值
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
    """
     提取核心子图：
    1. 删除权重最高的 remove_ratio 边；
    2. 找出所有非孤立节点；
    3. 将孤立节点按原始图中的度数排序，选出前若干名，使核心节点数达到原图节点数的一半；
    4. 用这些核心节点构造诱导子图，返回。
    """

    original_G = G.copy()
    total_nodes = original_G.number_of_nodes()
    target_core_size = total_nodes // 2
    print(f"目标核心节点数：{target_core_size}")

    # === 删除权重最大的边 ===
    all_edges = list(G.edges(data=True))
    sorted_edges = sorted(all_edges, key=lambda x: x[2].get(weight, 0), reverse=True)
    num_edges_to_remove = int(len(sorted_edges) * remove_ratio)
    edges_to_remove = [(u, v) for (u, v, d) in sorted_edges[:num_edges_to_remove]]
    G.remove_edges_from(edges_to_remove)
    print(f"[手术阶段] 实际删除边数：{num_edges_to_remove}")
    print(f"[手术阶段] 删除边后剩余边数：{G.number_of_edges()}")

    # === 获取非孤立节点和孤立节点 ===
    isolated_nodes = list(nx.isolates(G))
    non_isolated_nodes = [node for node in G.nodes if node not in isolated_nodes]
    print(f"非孤立节点数：{len(non_isolated_nodes)}")
    print(f"孤立节点数：{len(isolated_nodes)}")

    # === 计算孤立节点在原图中的度数并排序 ===
    isolated_degrees = {node: original_G.degree(node) for node in isolated_nodes}
    sorted_isolated = sorted(isolated_degrees.items(), key=lambda x: x[1], reverse=True)

    # === 从孤立节点中选择补足核心集所需的节点 ===
    num_to_add = target_core_size - len(non_isolated_nodes)
    top_isolated_nodes = [node for node, deg in sorted_isolated[:max(num_to_add, 0)]]

    core_nodes = set(non_isolated_nodes) | set(top_isolated_nodes)
    print(f"最终核心节点数：{len(core_nodes)}")

    # === 构造核心子图并复制边属性 ===
    core_subgraph = original_G.subgraph(core_nodes).copy()
    print(f"核心子图节点数：{core_subgraph.number_of_nodes()}，边数：{core_subgraph.number_of_edges()}")

    return core_subgraph


def build_residual_subgraph_from_core_nodes(original_G: nx.Graph, core_nodes: set):
    residual_nodes = set(original_G.nodes) - core_nodes
    residual_subgraph = original_G.subgraph(residual_nodes).copy()
    print(f"构建残余子图，节点数：{residual_subgraph.number_of_nodes()}，边数：{residual_subgraph.number_of_edges()}")
    return residual_subgraph


def compute_core_graph_metrics(G: nx.Graph, core_G: nx.Graph):
    core_nodes = set(core_G.nodes)
    residual_nodes = set(G.nodes) - core_nodes

    print("\n===== [1] 核心未加权凝聚力 r_d 计算过程 =====")

    def unweighted_degree(graph, node):
        return len(list(graph.edges(node)))

    numerator = 0.0
    core_node_count = len(core_nodes)

    for node in core_nodes:
        deg_core = unweighted_degree(core_G, node)
        deg_full = unweighted_degree(G, node)
        if deg_full > 0:
            ratio = deg_core / deg_full
            numerator += ratio
            # print(f"[DETAIL] Node {node}: deg_core = {deg_core}, deg_full = {deg_full}, ratio = {ratio:.4f}")
        else:
            print(f"[WARNING] Node {node} has deg_full = 0")

    r_d = numerator / core_node_count if core_node_count > 0 else 0.0
    print(f"[DEBUG] numerator = {numerator:.4f}, core_node_count = {core_node_count}, r_d = {r_d:.4f}")

    print("\n===== [2] 残余图结构碎片化指标 =====")
    G_res = build_residual_subgraph_from_core_nodes(G, core_nodes)

    print(f"[DEBUG] 残余节点数: {len(residual_nodes)}")
    print(f"[DEBUG] 残余子图节点数: {G_res.number_of_nodes()}")
    print(f"[DEBUG] 残余子图边数: {G_res.number_of_edges()}")

    # (a) runconnected
    print("---- (a) runconnected: 不连通节点对比例 ----")
    residual_node_list = list(residual_nodes)
    n = len(residual_node_list)
    if n < 2:
        runconnected = 0.0
        print("[DEBUG] 残余节点不足两个，runconnected = 0.0")
    else:
        unreachable_count = 0
        total_pairs = n * (n - 1) // 2
        component_map = {}
        for comp in nx.connected_components(G_res):
            for node in comp:
                component_map[node] = comp

        for i in range(n):
            for j in range(i + 1, n):
                u, v = residual_node_list[i], residual_node_list[j]
                if u not in component_map or v not in component_map or component_map[u] != component_map[v]:
                    unreachable_count += 1
                    # print(f"[DETAIL] ({u}, {v}) 不连通")

        runconnected = unreachable_count / total_pairs if total_pairs > 0 else 0.0
        print(f"[DEBUG] 不连通点对数: {unreachable_count}, 总点对数: {total_pairs}, runconnected = {runconnected:.4f}")

    # (b) r_k 分布
    print("---- (b) r_k: 边数分布 ----")
    components_res = list(nx.connected_components(G_res))
    total_res = len(components_res)
    print(f"[DEBUG] 残余图连通分量数: {total_res}")

    r_k_counter = Counter()
    for idx, comp in enumerate(components_res):
        subgraph = G_res.subgraph(comp)
        e_k = subgraph.number_of_edges()
        # print(f"[DETAIL] Component {idx}: 节点数 = {len(comp)}, 边数 = {e_k}")
        r_k_counter[e_k] += 1

    r_k = {k: count / total_res for k, count in sorted(r_k_counter.items())} if total_res > 0 else {}
    print(f"[DEBUG] r_k 分布: {r_k}")

    print("\n===== [3] 平均距离伸展 r_dist_stretch =====")
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
                stretch = dist_Gres / dist_G
                stretch_sum += stretch
                connected_pair_count += 1
                # print(f"[DETAIL] ({u}, {v}): dist_G = {dist_G}, dist_Gres = {dist_Gres}, stretch = {stretch:.4f}")
            except KeyError:
                # print(f"[SKIP] ({u}, {v}) 不连通，跳过")
                continue

    r_dist_stretch = stretch_sum / connected_pair_count if connected_pair_count > 0 else float('inf')
    print(f"[DEBUG] 成功配对数: {connected_pair_count}, r_dist_stretch = {r_dist_stretch:.4f}")

    print("\n===== [4] 核心图结构碎片化分布 r_core_k =====")
    components_core = list(nx.connected_components(core_G))
    total_core = len(components_core)
    print(f"[DEBUG] 核心图连通分量数: {total_core}")

    r_core_counter = Counter()
    for idx, comp in enumerate(components_core):
        subgraph = core_G.subgraph(comp)
        e_k = subgraph.number_of_edges()
        print(f"[DETAIL] Component {idx}: 节点数 = {len(comp)}, 边数 = {e_k}")
        r_core_counter[e_k] += 1

    r_core_k = {k: count / total_core for k, count in sorted(r_core_counter.items())} if total_core > 0 else {}
    print(f"[DEBUG] r_core_k 分布: {r_core_k}")

    return r_d, runconnected, r_k, r_dist_stretch, r_core_k


def print_core_metrics(r_d, runconnected, r_k, r_dist_stretch, r_core_k):
    print(f"核心加权凝聚力 r_d = {r_d:.4f}")
    print(f"参与图断连比例 runconnected= {runconnected:.4f}")
    print(f"剩余图结构碎片化分布 r_k = {r_k}")
    print(f"剩余图平均距离伸展 r_dist_stretch = {r_dist_stretch:.4f}")
    print(f"核心图结构碎片化分布 r_core_k = {r_core_k}")


# ===== 提取核心节点 =====
def extract_core_nodes(G: nx.Graph, method: str, core_size: int) -> set:
    """
    根据指定方法提取核心节点集合，并返回最大连通分支作为最终核心子图节点。
    支持方法：
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

    # 取 top-k 中心性节点
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:core_size]]

    # 构造核心子图
    core_subgraph = G.subgraph(top_nodes)

    # 提取最大连通分支
    largest_cc = max(nx.connected_components(core_subgraph), key=len)
    return set(largest_cc)



def main():
    # === 加载图 ===
    # citeseer数据集
    G = nx.read_gexf(r"C:\Users\22118\Downloads\citeseer_rf_sinkhorn_e2_20.gexf")
    # cora数据集
    # G =nx.read_gexf(r"C:\Users\22118\Downloads\cora_rf_sinkhorn_e2_20.gexf")
    # bio数据集
    # edge_file = r"C:\Users\22118\AppData\Local\Temp\0c9fa094-c7e4-4d94-8978-f81238834841_bio-CE-HT.zip.841\bio-CE-HT.edges"  # 替换为你的文件路径
    # df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target', 'weight'])
    # G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight')


    print(f"原始图节点数：{G.number_of_nodes()}, 边数：{G.number_of_edges()}")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"最大连通分支节点数：{G.number_of_nodes()}, 边数：{G.number_of_edges()}")
    else:
        print("图已经是连通的，无需提取最大连通分支。")

    G_original = copy.deepcopy(G)

    # === Ricci Flow 核心提取 ===
    rho_curv = RhoCurvature(copy.deepcopy(G), alpha=0.1, weight="weight", proc=cpu_count())
    print("计算初始Ricci曲率...")
    rho_curv.compute_ricci_curvature()
    print("开始Ricci流演化...")
    rho_curv.ricci_flow_iteration(iterations=12, step=0.1, delta=1e-6)

    print("开始手术操作...")
    core_subgraph_raw = extract_core_subgraph(copy.deepcopy(rho_curv.G), weight=rho_curv.weight, remove_ratio=0.8)
    largest_component = max(nx.connected_components(core_subgraph_raw), key=len)
    core_subgraph_largest = core_subgraph_raw.subgraph(largest_component).copy()
    core_nodes_largest = set(core_subgraph_largest.nodes())

    print("\n==== Ricci流生成的核心图（最大连通分量） ====")
    print(f"核心节点数: {len(core_nodes_largest)}")
    print(f"核心边数: {core_subgraph_largest.number_of_edges()}")
    print_core_metrics(*compute_core_graph_metrics(G_original, core_subgraph_largest))

    # === 其他方法的核心对比 ===
    core_sizes = {
        'degree': 416,
        'betweenness': 345,
        'closeness': 343,
        'pagerank': 484
    }

    core_nodes_all = {'Ricci Flow': core_nodes_largest}

    methods = ['degree', 'betweenness', 'closeness', 'pagerank']
    for method in methods:
        core_size = core_sizes.get(method, len(core_nodes_largest))
        print(f"\n==== 基于 {method} 方法提取核心子图（节点数: {core_size}） ====")
        core_nodes = extract_core_nodes(G_original, method, core_size)

        # 提取最大连通子图
        core_subgraph = G_original.subgraph(core_nodes)
        largest_cc = max(nx.connected_components(core_subgraph), key=len)
        core_nodes_largest_method = set(largest_cc)
        core_subgraph_largest_method = core_subgraph.subgraph(largest_cc).copy()

        core_nodes_all[method.capitalize()] = core_nodes_largest_method

        print(
            f"核心节点数: {len(core_nodes_largest_method)}, 核心边数: {core_subgraph_largest_method.number_of_edges()}")
        print_core_metrics(*compute_core_graph_metrics(G_original, core_subgraph_largest_method))




if __name__ == "__main__":
    main()


