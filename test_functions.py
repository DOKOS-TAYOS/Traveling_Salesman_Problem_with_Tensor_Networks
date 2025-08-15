import torch
from time import perf_counter
from IPython.display import display
from tsp_module import tn_tsp_solver
from auxiliary_solver import ortools_tsp_solver
from auxiliary_functions import cost_solution, has_repeated_elements

def solve_tsp_full_with_tn(distances_matrix: torch.Tensor, tau: float):
    """
    Solve TSP using tensor network with full computation.
    
    Args:
        distances_matrix: Tensor of distances between cities
        tau: Temperature parameter for the solver
        
    Returns:
        Tuple of (solution, cost, has_repeated_cities)
    """
    solution = tn_tsp_solver(distances_matrix, tau, verbose=True, n_layers=None)
    print('Tensor Network solution: '+', '.join(map(str, solution.tolist())))

    repeated = has_repeated_elements(solution)
    if repeated:
        print('The solution has repeated cities.')

    cost = cost_solution(solution, distances_matrix)
    print(f'Tensor Network cost: {cost}')

    return solution, cost, repeated

def solve_tsp_truncated_with_tn(distances_matrix: torch.Tensor, tau: float, n_layers: int):
    """
    Solve TSP using tensor network with truncated computation.
    
    Args:
        distances_matrix: Tensor of distances between cities
        tau: Temperature parameter for the solver
        n_layers: Number of layers to use in the tensor network
        
    Returns:
        Tuple of (solution, cost, has_repeated_cities)
    """
    solution = tn_tsp_solver(distances_matrix, tau, verbose=True, n_layers=n_layers)
    print('Tensor Network solution: '+', '.join(map(str, solution.tolist())))

    repeated = has_repeated_elements(solution)
    if repeated:
        print('The solution has repeated cities.')

    cost = cost_solution(solution, distances_matrix)
    print(f'Tensor Network cost: {cost}')

    return solution, cost, repeated

def tn_vs_ortools(distances_matrix: torch.Tensor, tau: float, n_layers: int, verbose: bool = True):
    """
    Compare tensor network and OR-Tools solutions for TSP.
    
    Args:
        distances_matrix: Tensor of distances between cities
        tau: Temperature parameter for the tensor network solver
        n_layers: Number of layers to use in the tensor network
        verbose: Whether to print detailed output
        
    Returns:
        Tuple of (tn_solution, tn_cost, tn_has_repeated, ortools_solution, ortools_cost, ortools_has_repeated)
    """
    PENALTY = 1000  # Penalty for failed solutions
    PROPORTION = 2

    # Solve with tensor network
    start = perf_counter()
    solution_tn = tn_tsp_solver(distances_matrix, tau, verbose=verbose, n_layers=n_layers)
    time_limit = PROPORTION*int(perf_counter() - start)
    # Solve with OR-Tools
    solution_ortools = ortools_tsp_solver(distances_matrix, verbose=False, time_limit=time_limit)

    repeated_tn = has_repeated_elements(solution_tn)
    cost_tn = cost_solution(solution_tn, distances_matrix)

    
    if solution_ortools is not None:
        cost_ortools = cost_solution(solution_ortools, distances_matrix)
        repeated_ortools = has_repeated_elements(solution_ortools)
    else:
        cost_ortools = PENALTY
        repeated_ortools = True
    
    if verbose:
        print('Tensor Network solution: '+', '.join(map(str, solution_tn.tolist())))
        if solution_ortools is not None:
            print('OR-Tools solution:       ' + ', '.join(map(str, solution_ortools.tolist())))

        if repeated_tn:
            print('The tensor network solution has repeated cities.')
        if repeated_ortools:
            print('The OR-Tools solution has repeated cities.')

        print(f'Tensor Network cost: {cost_tn}')
    
        if solution_ortools is not None:
            print(f'OR-Tools cost:       {cost_ortools}')
        else:
            print('OR-Tools failed to find a solution.')

    return solution_tn, cost_tn, repeated_tn, solution_ortools, cost_ortools, repeated_ortools