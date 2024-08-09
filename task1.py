import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import time


# 构建稀疏矩阵
def load_data(file_path):
    edges = []
    max_node = 0
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                from_node, to_node = map(int, line.split())
                edges.append((from_node, to_node))
                max_node = max(max_node, from_node, to_node)
    
    return edges, max_node + 1  # Nodes从0开始，需要+1


# 构建邻接矩阵
def build_adjacency_matrix(edges, num_nodes):
    data = np.ones(len(edges))
    row_indices = [to_node for _, to_node in edges]
    col_indices = [from_node for from_node, _ in edges]
    
    # 临街矩阵
    M = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    
    # 按列正则
    M = M.multiply(1.0 / M.sum(axis=0).A.ravel())  
    
    return M



# 初始化排序向量r
def initialize_rank_vector(num_nodes):
    return np.ones(num_nodes) / num_nodes



# 幂迭代
def power_iteration(M, r, epsilon=0.02, max_iterations=1000):
    for i in range(max_iterations):
        r_next = M @ r
        # L1正则停止条件
        if np.linalg.norm(r_next - r, 1) < epsilon:
            print(f'Convergence reached after {i+1} iterations.')
            return r_next, i + 1
        r = r_next
    print('Max iterations reached without convergence.')
    return r, max_iterations



# pagerank
def run_pagerank(file_path):
    start_time = time.time()

    # 创建邻接矩阵
    edges, num_nodes = load_data(file_path)
    M = build_adjacency_matrix(edges, num_nodes)
    
    # 初始化r
    r = initialize_rank_vector(num_nodes)
    
    # 幂迭代
    r, iterations = power_iteration(M, r)
    
    # 输出结果
    end_time = time.time()
    print(f'1. Power iteration took {end_time - start_time:.4f} seconds.')
    print(f'2. Total iterations: {iterations}')
    
    top_10_indices = np.argsort(-r)[:10]
    print('3. Top 10 nodes by PageRank:')
    for i in top_10_indices:
        print(f'Node {i}: {r[i]}')


if __name__ == '__main__':
    run_pagerank('./files/web-Google.txt')
