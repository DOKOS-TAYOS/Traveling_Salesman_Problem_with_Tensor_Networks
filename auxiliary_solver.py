import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from auxiliary_functions import cost_solution

def create_data_model(distances: torch.Tensor, n_cities: int):
    """Stores the data for the problem.
    
    Args:
        distances: Tensor of distances between nodes.
        n_cities: Number of cities in the problem.
        
    Returns:
        dict: Data model containing distance matrix, number of vehicles, and depot.
    """
    PENALTY = 3000  # Large value to represent "infinity" or unreachable paths
    data = {}
    
    # Convert distance tensor to integer matrix, replacing infinity with a large penalty value
    data["distance_matrix"] = []
    for origin_list in distances:
        row = []
        for dist in origin_list:
            if torch.isinf(dist):
                row.append(PENALTY)
            else:
                # Handle both scalar tensors and tensors with dimensions
                row.append(int(dist.item()))
        data["distance_matrix"].append(row)
    
    data["num_vehicles"] = 1  # Single vehicle for TSP
    data["depot"] = n_cities - 1  # Starting point
    return data


def print_solution(manager, routing, solution, distances):
    """
    Prints the solution on console with formatted output.
    
    Args:
        manager: Routing index manager
        routing: Routing model
        solution: Solution found by the solver
        distances: Tensor of distances between nodes
    """
    # Get objective value
    objective_value = solution.ObjectiveValue()
    print(f"Objective: {objective_value} miles")
    
    # Extract route information
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    route = [index]
    
    # Build route path
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        plan_output += f" {node} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route.append(index)
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    
    # Add final node to output
    plan_output += f" {manager.IndexToNode(index)}\n"
    plan_output += f"Route distance: {route_distance} miles\n"
    
    # Print formatted output
    print(chr(27) + "[1;37m" + plan_output)
    
    # Print route and calculated cost
    route_nodes = route[:-1]  # Exclude the last node which is a duplicate of the first
    print(route_nodes)
    print(chr(27) + "[1;36m" + f'Google solution cost: {cost_solution(route_nodes, distances)}')

















def ortools_tsp_solver(distances: torch.Tensor, verbose: bool=False, time_limit: int=100):
    """
    Solves the Travelling Salesperson Problem (TSP) using Google OR-Tools.
    
    Args:
        distances: Tensor of distances between nodes
        verbose: If want messages
        
    Returns:
        The solution route as a torch.Tensor if found, None otherwise
    """
    # Get number of cities from the distances tensor
    n_cities = distances.size(0)
    
    # Instantiate the data problem
    data = create_data_model(distances, n_cities)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback function
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # Register and set the transit callback
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Configure solver settings
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # search_parameters.guided_local_search_lambda_coefficient = 0.05
    
    # Add time limit for large problems
    search_parameters.time_limit.seconds = time_limit
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Process and return the solution
    if solution:
        if verbose:
            print_solution(manager, routing, solution, distances)
        
        # Extract the route for returning
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return torch.tensor(route, dtype=torch.long)
    else:
        if verbose:
            print("No solution found!")
        return None
