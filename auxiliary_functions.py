import numpy as np
import torch
from IPython.display import HTML

def generate_problem(n_cities: int, n_connections: int, distance_range: float) -> torch.Tensor:
    """
    Generate a TSP problem instance.
    
    Args:
        n_cities: Number of cities in the problem.
        n_connections: Maximum number of outgoing connections per city.
        distance_range: Maximum distance between cities (distances will be in range [0, distance_range]).
    
    Returns:
        A tensor of shape (n_cities, n_cities) representing the distance matrix,
        where distances[i, j] is the distance from city i to city j.
        Unreachable cities have a distance of infinity.
    """
    # Initialize distance matrix with infinity
    distances = torch.ones(size=(n_cities, n_cities), dtype=torch.int32) * float('inf')
    
    for city in range(n_cities):
        # Get all other cities
        other_cities = torch.arange(n_cities, dtype=torch.int32)
        other_cities = other_cities[other_cities != city]
        
        # Randomly shuffle to select random connections
        other_cities = other_cities[torch.randperm(len(other_cities))]
        
        # Connect to at most n_connections other cities
        for i in range(min(n_connections, len(other_cities))):
            destination = other_cities[i]
            # Generate random distance in [0, distance_range]
            distances[city, destination] = torch.rand(1) * distance_range
    
    return distances

def cost_solution(solution: torch.Tensor, distances: torch.Tensor) -> float:
    """
    Calculate the total cost of a TSP solution given a distance matrix.
    
    Args:
        solution: A tensor containing the ordered sequence of cities to visit.
        distances: A tensor of shape (n_cities, n_cities) representing the distance matrix.
    
    Returns:
        The total cost of the tour, including the return to the starting city.
    """
    # Calculate costs between consecutive cities in the path
    path_costs = distances[solution[:-1], solution[1:]]
    
    # Add the cost of returning to the starting city
    return_cost = distances[solution[-1], solution[0]]
    
    # Sum all costs
    total_cost = torch.sum(path_costs) + return_cost
    
    return total_cost.item() if torch.is_tensor(total_cost) else total_cost

def has_repeated_elements(solution: torch.Tensor) -> bool:
    """
    Check if there are any repeated elements in the solution.
    
    Args:
        solution: A tensor containing the ordered sequence of cities to visit.
    
    Returns:
        True if there are repeated elements, False otherwise.
    """
    # Convert to a set and compare lengths
    unique_elements = torch.unique(solution)
    return len(unique_elements) < len(solution)

def progress(value: int, max_value: int=100):
    """
    Function that displays a progress bar with percentage.
    
    Args:
        value: Current progress value.
        max_value: Maximum value representing 100% completion (default: 100).
    
    Returns:
        HTML progress bar with percentage display.
    """
    percentage = np.round((value/max_value*100), decimals=2)
    
    return HTML("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <progress 
                value='{value}' 
                max='{max_value}' 
                style='width: 70%; margin-right: 10px;'
            ></progress>
            <span>{percentage:.2f}%</span>
        </div>
        """.format(value=value, max_value=max_value, percentage=percentage))