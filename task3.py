import numpy as np
from scipy.sparse import csr_matrix
import time
from task1 import load_data, build_adjacency_matrix, initialize_rank_vector


# 幂迭代引入teleport
def power_iteration_with_teleport(M, r, beta=0.9, epsilon=0.02, max_iterations=1000):
    N = M.shape[0]
    for i in range(max_iterations):
        r_new = beta * (M @ r)
        S = np.sum(r_new)  # 计算常数 S
        r_new = r_new + (1 - S) / N
        
        if np.linalg.norm(r_new - r, 1) < epsilon:
            print(f'Convergence reached after {i+1} iterations.')
            return r_new, i + 1
        r = r_new
    
    print('Max iterations reached without convergence.')
    return r, max_iterations

# 有teleport的pagerank
def run_pagerank_with_teleport(file_path, beta=0.9):
    start_time = time.time()

    # 创建邻接矩阵
    edges, num_nodes = load_data(file_path)
    M = build_adjacency_matrix(edges, num_nodes)
    
    # 初始化r
    r = initialize_rank_vector(num_nodes)
    
    # 幂迭代
    r, iterations = power_iteration_with_teleport(M, r, beta=beta)
    
    # 输出结果
    end_time = time.time()
    print(f'1. Power iteration with teleport (beta={beta}) took {end_time - start_time:.4f} seconds.')
    print(f'2. Total iterations: {iterations}')
    
    top_10_indices = np.argsort(-r)[:10]
    print('3. Top 10 nodes by PageRank:')
    for i in top_10_indices:
        print(f'Node {i}: {r[i]}')

# 变更beta值的rankpage结果
def run_experiments_with_varied_beta(file_path):
    betas = [1, 0.9, 0.8, 0.7, 0.6]
    for beta in betas:
        print(f'\nRunning PageRank with beta={beta}')
        run_pagerank_with_teleport(file_path, beta=beta)


if __name__ == '__main__':
    run_experiments_with_varied_beta('./files/web-Google.txt')
