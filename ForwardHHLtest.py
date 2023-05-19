import numpy as np
from linear_solvers import NumPyLinearSolver, HHL
from qiskit.quantum_info import Statevector
matrix = np.array([[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0,],[0,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]])
vector=np.array([[0.73],[0.548],[0.365],[0.183],[0],[0],[0],[0]])
naive_hhl_solution = HHL().solve(matrix, vector)
classical_solution = NumPyLinearSolver().solve(matrix,vector/np.linalg.norm(vector))
print(naive_hhl_solution.state)
#把向量取出来
naive_sv = Statevector(naive_hhl_solution.state).data
#只取第一个比特为1的向量
naive_full_vector = np.array([[naive_sv[260]],[naive_sv[261]],[naive_sv[262]] ,[naive_sv[263]]])
#结果去虚数
def get_solution_vector(solution):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data[260:264].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)
print('full naive solution vector:', get_solution_vector(naive_hhl_solution))
print('classical state:', classical_solution.state)
