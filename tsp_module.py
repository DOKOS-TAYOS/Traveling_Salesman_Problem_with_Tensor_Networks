import torch
import tensorkrowch as tk
from auxiliary_functions import progress
from IPython.display import display

def generate_superposition_layer(tn: tk.TensorNetwork, n_nodes: int) -> list[tk.Node]:
    """
    Creates uniform superposition vectors of cities.
    Creates a template tensor and makes all other tensors point to this template.
    
    Args:
        tn: Tensor network to store the tensors.
        n_nodes: Number of tensors to generate.
        
    Returns:
        layer: List of tensors in the layer.
    """
    # Create template tensor with uniform distribution
    uniform_node = tk.Node(tensor=torch.ones(n_nodes), name='uniform', axes_names=['right'], network=tn, virtual=True)

    # Initialize list for the different tensors
    layer = []
    
    # Create nodes and set them to point to the uniform template
    for i in range(n_nodes):
        layer.append(tk.Node(shape=(n_nodes,), name=f'initial_({i},{0})', axes_names=['right'], network=tn))
        layer[i].set_tensor_from(uniform_node)

    return layer

def generate_evolution_layer(tn: tk.TensorNetwork, n_nodes: int, distances: torch.Tensor, tau: float) -> list[tk.Node]:
    """
    Creates the MPO tensors responsible for imaginary time evolution.
    Creates templates and makes other tensors point to these template tensors for efficiency.
    
    Args:
        tn: Tensor network to store the tensors.
        n_nodes: Number of tensors to generate.
        distances: Tensor of distances between nodes. distances[i,j] is the distance from node i to node j.
                  We want to travel from the penultimate node to the last node.
        tau: Damping constant for imaginary time evolution.

    Returns:
        layer: List of tensors in the evolution layer.
    """
    # Initialize the list for evolution layer nodes
    layer = []
    
    # Create the tensor for the initial node
    initial_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes))
    for current in range(n_nodes):
        # Calculate the negative exponential of the distance multiplied by tau
        initial_tensor[current, current, current] = torch.exp(-tau * distances[n_nodes, current])

    # Create the initial node and add it to the layer
    initial_node = tk.Node(
        tensor=initial_tensor, 
        name=f'evolution_(0,1)', 
        network=tn, 
        axes_names=['left', 'right', 'down']
    )
    layer.append(initial_node)

    # Create the tensor for intermediate nodes
    intermediate_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes, n_nodes))
    for previous in range(n_nodes):
        for current in range(n_nodes):
            # Calculate the negative exponential of the distance multiplied by tau
            intermediate_tensor[current, current, previous, current] = torch.exp(-tau * distances[previous, current])

    # Create a virtual template node with the intermediate tensor
    intermediate_node = tk.Node(
        tensor=intermediate_tensor, 
        name='evolution_(uniform)', 
        network=tn, 
        virtual=True,
        axes_names=['left', 'right', 'up', 'down']
    )

    # Create the intermediate nodes and add them to the layer
    for node in range(1, n_nodes-1):
        layer.append( tk.Node(
            shape=(n_nodes, n_nodes, n_nodes, n_nodes), 
            name=f'evolution_({node},1)', 
            network=tn, 
            axes_names=['left', 'right', 'up', 'down']
        ) )
        layer[node].set_tensor_from(intermediate_node)
        
        # Connect to previous node
        layer[node]['up'] ^ layer[node-1]['down']

    # Create the tensor for the final node
    final_tensor = torch.zeros(size=(n_nodes, n_nodes, n_nodes))
    for previous in range(n_nodes):
        for current in range(n_nodes):
            # Calculate the negative exponential of the sum of distances multiplied by tau
            # Add the connection to the last city
            final_tensor[current, current, previous] = torch.exp(-tau * (
                distances[previous, current] + distances[current, n_nodes+1]
            ))

    # Create the final node and add it to the layer
    layer.append( tk.Node(
        tensor=final_tensor, 
        name=f'evolution_({n_nodes-1},1)', 
        network=tn, 
        axes_names=['left', 'right', 'up']
    ) )

    # Connect final node to the previous one
    layer[-1]['up'] ^ layer[-2]['down']

    return layer

def generate_restriction_layer(tn: tk.TensorNetwork, n_nodes: int, target_node: int) -> list[tk.Node]:
    """
    Function that creates the MPO tensors responsible for enforcing city visit restrictions.
    Creates a template tensor and makes all other tensors point to this template for efficiency.
    
    Args:
        tn: Tensor network to store the tensors.
        n_nodes: Number of tensors to generate.
        target_node: City that cannot be repeated.

    Returns:
        layer: List of tensors in the restriction layer.
    """
    # Initialize the list for restriction layer nodes
    layer = []

    # Create the tensor for the initial node
    initial_tensor = torch.zeros(size=(n_nodes, n_nodes, 2))
    for current in range(n_nodes):
        # Set appropriate state based on whether current city is the target
        initial_tensor[current, current, int(current == target_node)] = 1

    # Create the initial node and add it to the layer
    layer.append(tk.Node(
        tensor=initial_tensor, 
        name=f'restr_({target_node},{0})', 
        network=tn, 
        axes_names=['left', 'right', 'down']
    ))

    # Create the tensor for intermediate nodes
    intermediate_tensor = torch.zeros(size=(n_nodes, n_nodes, 2, 2))
    for current in range(n_nodes):
        if current == target_node:
            # If current city is the restricted one, only allow transition from unvisited to visited
            intermediate_tensor[current, current, 0, 1] = 1
        else:
            # For other cities, allow both states to persist
            intermediate_tensor[current, current, 0, 0] = 1
            intermediate_tensor[current, current, 1, 1] = 1

    # Create a virtual template node with the intermediate tensor
    intermediate_node = tk.Node(
        tensor=intermediate_tensor, 
        name=f'restr_({target_node},uniform)', 
        network=tn, 
        virtual=True,
        axes_names=['left', 'right', 'up', 'down']
    )

    # Create the intermediate nodes and add them to the layer
    for i_node in range(1, n_nodes-1):
        layer.append( tk.Node(
            shape=(n_nodes, n_nodes, 2, 2), 
            name=f'restr_({target_node},{i_node})', 
            network=tn, 
            axes_names=['left', 'right', 'up', 'down']
        ) )
        layer[i_node].set_tensor_from(intermediate_node)

        # Connect to previous node
        layer[i_node]['up'] ^ layer[i_node-1]['down']

    # Create the tensor for the final node
    final_tensor = torch.zeros(size=(n_nodes, n_nodes, 2))
    for current in range(n_nodes):
        # For the final node, target city must be visited (state 0), others must not be (state 1)
        final_tensor[current, current, int(current != target_node)] = 1

    # Create the final node and add it to the layer
    layer.append( tk.Node(
        tensor=final_tensor, 
        name=f'restr_({target_node},{n_nodes-1})', 
        network=tn, 
        axes_names=['left', 'right', 'up']
    ) )

    # Connect final node to the previous one
    layer[-1]['up'] ^ layer[-2]['down']

    return layer


def generate_trace_layer(tn: tk.TensorNetwork, n_nodes: int) -> list[tk.Node]:
    """
    Function that creates a layer of trace tensors for the final contraction.
    Creates a template tensor of ones and makes all other tensors point to this template.
    
    Args:
        tn: Tensor network where these tensors will be stored.
        n_nodes: Number of tensors to generate.
        
    Returns:
        layer: List of trace tensors in the layer.
    """
    # Create a virtual template node with ones
    uniform_node = tk.Node(
        tensor=torch.ones(n_nodes), 
        name='trace_template', 
        axes_names=['left'], 
        network=tn, 
        virtual=True
    )

    # Initialize the layer list with list comprehension for efficiency
    layer = []
    for i in range(n_nodes):
        layer.append(tk.Node(
            shape=(n_nodes,), 
            name=f'trace_({i})', 
            axes_names=['left'], 
            network=tn
        ))
        layer[i].set_tensor_from(uniform_node)

    return layer



def create_tensors(tn: tk.TensorNetwork, distances: torch.Tensor, tau: float, n_layers: int|None) -> list[list[tk.Node]]:
    """
    Creates the tensors of the tensor network for solving the TSP problem.
    
    Args:
        tn: Tensor network to store the tensors.
        distances: Tensor of distances between nodes. distances[i,j] is the distance from node i to node j.
                  We want to travel from the penultimate node to the last node.
        tau: Damping constant for imaginary time evolution.
        n_layers: Maximum number of restriction layers to apply. If None, all possible restrictions are applied.
        
    Returns:
        A list of tensor layers, where list[i][j] refers to tensor j of layer i.
    """
    # Number of nodes in the subproblem
    n_nodes = len(distances) - 2

    # Initialization layer
    superp_layer = generate_superposition_layer(tn, n_nodes)

    # Imaginary time evolution layer
    evol_layer = generate_evolution_layer(tn, n_nodes, distances, tau)

    # Restriction layer
    restr_layer = []
    
    # Determine which restriction layers to apply
    if n_layers is None or n_layers >= n_nodes - 1:  # Apply all possible restrictions
        target_nodes = range(n_nodes - 1)
    else:  # Randomly select which restrictions to apply
        target_nodes = torch.randperm(n_nodes - 1)[:n_layers]
    
    # Generate the selected restriction layers
    for target_node in target_nodes:
        restr_layer.append(generate_restriction_layer(tn, n_nodes, target_node))

    # Tracing layer
    trace_layer = generate_trace_layer(tn, n_nodes)

    # Connect the tensors
    for node in range(n_nodes):
        # Connect superposition layer to evolution layer
        superp_layer[node]['right'] ^ evol_layer[node]['left']
        
        if not restr_layer:  # If there are no restrictions
            # Connect evolution layer directly to trace layer
            evol_layer[node]['right'] ^ trace_layer[node]['left']
        else:  # If there are restrictions
            # Connect evolution layer to first restriction layer
            evol_layer[node]['right'] ^ restr_layer[0][node]['left']
            
            # Connect restriction layers to each other
            for depth in range(len(restr_layer) - 1):
                restr_layer[depth][node]['right'] ^ restr_layer[depth + 1][node]['left']
            
            # Connect final restriction layer to trace layer
            restr_layer[-1][node]['right'] ^ trace_layer[node]['left']

    # Build and return the complete tensor network as a list of layers
    layers = [superp_layer, evol_layer]
    layers.extend(restr_layer)
    layers.append(trace_layer)
    
    return layers



def contract_tensors(tensor_network: list[list[tk.Node]]) -> torch.Tensor:
    """
    Efficiently contracts a tensor network representing a TSP problem.
    
    Args:
        tensor_network: List of layers, where each layer is a list of tensors.
                       tensor_network[i][j] represents tensor j of layer i.
    
    Returns:
        A flattened tensor representing the result of the contraction.
    """
    # Get dimensions of the network
    n_qudits = len(tensor_network[0])
    depth = len(tensor_network)

    # Contract the superposition layer with the evolution layer
    for qudit in range(n_qudits):
        tensor_network[1][qudit] = tk.contract_between_(tensor_network[0][qudit], tensor_network[1][qudit])
        tensor_network[1][qudit].name = f'Initial_evolution_({qudit})'

    # Contract the tracing layer with the last restriction layer
    for qudit in range(1, n_qudits):
        tensor_network[-2][qudit] = tk.contract_between_(tensor_network[-1][qudit], tensor_network[-2][qudit])
        tensor_network[-2][qudit].name = f'trace_restr_({qudit})'

    # Start with the last qudit in the second-to-last layer
    result = tensor_network[-2][-1]

    # Contract the last qudit through all layers from end to beginning
    for layer in range(depth-3, 0, -1):
        result = tk.contract_between_(result, tensor_network[layer][-1])
    
    # Contract remaining qudits
    for qudit in range(n_qudits-2, -1, -1):
        for layer in range(1, depth-1):
            result = tk.contract_between_(result, tensor_network[layer][qudit])

    return result.tensor.flatten()



def TSP_solver_from_to(distances: torch.Tensor, tau: float, n_layers: int|None) -> torch.Tensor:
    """
    Solves the partial Traveling Salesman Problem using tensor networks.
    
    Args:
        distances: Tensor of distances between nodes. distances[i,j] is the distance from node i to node j.
                  We want to travel from the penultimate node to the last node.
        tau: Damping constant for imaginary time evolution.
        n_layers: Maximum number of restriction layers to apply. None applies all the restrictions.
        
    Returns:
        vector_solution: Tensor resulting from contracting the tensor network, containing
                        the probability amplitudes for each possible first node in the route.
    """
    # Create the tensor network
    tn = tk.TensorNetwork(name='TSP')

    # Create the tensors for the network
    tensor_network = create_tensors(tn, distances, tau, n_layers)

    # Contract the tensor network to obtain the solution
    vector_solution = contract_tensors(tensor_network)
    
    return vector_solution



def tn_tsp_solver(distances: torch.Tensor, tau: float = 1, verbose: bool = True, n_layers: int|None = None) -> torch.Tensor:
    """
    General function for solving the TSP using the iterative method.
    
    This solver uses an iterative approach to determine the optimal route. Due to the 
    symmetry of the problem, we designate the last node as the first in our route to 
    simplify the algorithm.
    
    The algorithm works as follows:
    1. We fix the first node of the solution as N-1 (the last node).
    2. We modify the distance matrix so no node can travel to the already selected node.
    3. We move the row and column of the selected node to the penultimate position.
    4. We solve the partial TSP problem, which finds the shortest route from the 
       penultimate node (departure) to the last node (destination).
    5. After determining the next node in the route, we remove its row and column from 
       the distance matrix and move the newly selected node to the penultimate position.
    6. We repeat steps 3-5 until all nodes are determined.
    
    To reduce computational complexity, we limit the number of active restriction layers 
    to n_layers throughout the process.
    
    Args:
        distances: Tensor of distances between nodes. distances[i,j] is the distance 
                  from node i to node j.
        tau: Damping constant for imaginary time evolution.
        verbose: Whether to display a progress bar.
        n_layers: Maximum number of restriction layers to apply. None applies all the restrictions
        
    Returns:
        solution: Tensor containing the route of nodes in order of visit.
    """
    # Number of cities and number of remaining cities
    n_cities = distances.shape[0]
    remaining_cities = torch.arange(n_cities - 1)

    # Initialize progress bar if verbose
    if verbose:
        out = display(progress(0, n_cities), display_id=True)
        var_counter = 0

    # Solution vector
    solution = torch.zeros(n_cities, dtype=torch.int64)
    solution[0] = n_cities - 1

    # Update progress bar
    if verbose:
        var_counter += 1
        out.update(progress(var_counter, n_cities))

    # Loop in which we will create each partial TSP and solve its first variable
    for t in range(1, n_cities-1):
        # Create new distance matrix for the partial problem
        new_distances = torch.full((n_cities - t + 2, n_cities - t + 2), float('inf'))
        
        # Fill distances between remaining cities
        for i, city_origin in enumerate(remaining_cities):
            for j, city_destination in enumerate(remaining_cities):
                new_distances[i, j] = distances[city_origin, city_destination]
            
            # Add distances to the final node
            new_distances[i, -1] = distances[city_origin, -1]

        # Add distances from the previously selected city to all remaining cities
        for j, city_destination in enumerate(remaining_cities):
            new_distances[-2, j] = distances[solution[t-1], city_destination]

        # Add distance from previous city to final city (should rarely be used)
        if t != 1:
            new_distances[-2, -1] = distances[solution[t-1], -1]

        # Solve the partial TSP problem
        vector_solution = TSP_solver_from_to(new_distances, tau, n_layers)

        # Check if we can no longer meet the restrictions
        if torch.max(vector_solution) < 1e-120:
            break

        # Get the best next city
        partial_solution = torch.argmax(vector_solution).item()

        # Map the partial solution to the global solution
        solution[t] = remaining_cities[partial_solution]

        # Remove the selected city from remaining cities
        remaining_cities = torch.cat([remaining_cities[:partial_solution], remaining_cities[partial_solution+1:]])

        # Update progress bar
        if verbose:
            var_counter += 1
            out.update(progress(var_counter, n_cities))

    # Add the last remaining city to complete the route
    solution[-1] = remaining_cities[0]
    
    # Final progress bar update
    if verbose:
        var_counter += 1
        out.update(progress(var_counter, n_cities))

    return solution


