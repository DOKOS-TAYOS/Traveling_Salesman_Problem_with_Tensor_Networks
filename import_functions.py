import json
import pandas as pd
import torch

def import_distance_matrix_from_csv(st, uploaded_file):
    """Import distance matrix from CSV file"""
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        # Ensure the matrix is square
        if df.shape[0] != df.shape[1]:
            st.error("Distance matrix must be square (same number of rows and columns)")
            return None, 0
        
        # Convert to tensor
        distance_matrix = torch.tensor(df.values, dtype=torch.float32)
        n_nodes = distance_matrix.shape[0]
        
        return distance_matrix, n_nodes
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None, 0

def import_distance_matrix_from_json(st, uploaded_file):
    """Import distance matrix from JSON file"""
    try:
        data = json.load(uploaded_file)
        
        # Check if it's a simple 2D array
        if isinstance(data, list) and all(isinstance(row, list) for row in data):
            matrix_data = data
        # Check if it's a dict with a 'distance_matrix' key
        elif isinstance(data, dict) and 'distance_matrix' in data:
            matrix_data = data['distance_matrix']
        else:
            st.error("JSON file must contain a 2D array or have a 'distance_matrix' key")
            return None, 0
        
        # Ensure the matrix is square
        if len(matrix_data) != len(matrix_data[0]) or not all(len(row) == len(matrix_data) for row in matrix_data):
            st.error("Distance matrix must be square (same number of rows and columns)")
            return None, 0
        
        # Convert to tensor
        distance_matrix = torch.tensor(matrix_data, dtype=torch.float32)
        n_nodes = distance_matrix.shape[0]
        
        return distance_matrix, n_nodes
    except Exception as e:
        st.error(f"Error reading JSON file: {str(e)}")
        return None, 0

def export_solution_as_json(solution, cost, runtime, distance_matrix):
    """Export solution as JSON"""
    solution_data = {
        "solution": {
            "route": [int(city.item()) for city in solution],
            "cost": float(cost),
            "runtime_seconds": float(runtime),
            "number_of_cities": len(solution)
        },
        "distance_matrix": distance_matrix.tolist(),
        "metadata": {
            "algorithm": "Tensor Network TSP Solver",
            "paper": "https://arxiv.org/abs/2311.14344"
        }
    }
    return json.dumps(solution_data, indent=2)