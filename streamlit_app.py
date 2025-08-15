import streamlit as st
import torch
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from time import perf_counter

from tsp_module import tn_tsp_solver
from auxiliary_functions import generate_problem, cost_solution, has_repeated_elements
from import_functions import import_distance_matrix_from_csv, import_distance_matrix_from_json, export_solution_as_json

# Page configuration
st.set_page_config(
    page_title="🧭 Tensor Network TSP Solver",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        font-style: italic;
        margin-bottom: 0.2rem;
    }
    .feature-box {
        background: #1e1e1e;
        padding: 0.3rem;
        border-radius: 3px;
        border-left: 1px solid #667eea;
        margin: 0.1rem 0;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem;
        border-radius: 3px;
        text-align: center;
        margin: 0.1rem 0;
    }
    .success-box {
        background: #143a1e;
        border: 1px solid #1e5a2d;
        color: #a0e8b7;
        padding: 0.2rem;
        border-radius: 2px;
        margin: 0.1rem 0;
    }
    .warning-box {
        background: #3a3000;
        border: 1px solid #5a4a00;
        color: #ffe066;
        padding: 0.2rem;
        border-radius: 2px;
        margin: 0.1rem 0;
    }
    .info-box {
        background: #002b36;
        border: 1px solid #004052;
        color: #8ec9d6;
        padding: 0.2rem;
        border-radius: 2px;
        margin: 0.1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.2rem 0.8rem;
        font-weight: bold;
        transition: transform 0.1s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown('<h1 class="main-header">🧭 Tensor Network Traveling Salesman Problem Solver</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🚀 A quantum-inspired algorithm using tensor networks for optimizing routes by Alejandro Mata Ali</p>', unsafe_allow_html=True)

# Enhanced About Section
with st.expander("📖 🎓 About the Algorithm and How to Use", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🏙️ The Traveling Salesman Problem (TSP)
        
        The TSP is one of the most famous optimization challenges in computer science. 🤔
        
        **🎯 Goal:** Find the shortest route that visits all cities exactly once and returns home.
        """)
        st.image("TSP_Nodes.png")
        st.markdown("""
        In this example, the challenge is to find the optimal path connecting all cities (nodes) while minimizing the total distance traveled. The blue lines represent possible connections between cities, with different distances, and red lines are a selected route.
        
        **💡 Why it matters:** Applications in logistics, manufacturing, DNA sequencing, and circuit design.

        ## 📐 Mathematical Formulation
        
        The algorithm minimizes the total distance traveled:
        
        $$C(\\vec{x}) = \sum_{t=0}^{n-1} T_{x_t, x_{t+1}}$$
        
        Where:
        - $x_t$ is the city in the time-step t
        - $T_{i,j}$ is the distance between cities $i$ and $j$
        - $x_{n} = x_0$ (return to starting city)
        
        **🔍 Constraints:**
        - Each city must be visited exactly once
        - The tour must form a complete cycle
        """)
    
    with col2:
        st.markdown("""
        
        ## ⚛️ Tensor Network Algorithm
        
        The tensor network algorithm implemented is based on the research presented in the paper ["Traveling Salesman Problem from a Tensor Networks Perspective"](https://arxiv.org/abs/2311.14344), Alejandro Mata Ali et al.
        The approach consists of a quantum-inspired construction that:

        
        **🔄 Step 1:** *Quantum Superposition* - All possible routes exist simultaneously
        
        **🌀 Step 2:** *Imaginary Time Evolution* - Shorter paths have exponentially larger probabilities
        
        **🛡️ Step 3:** *Constraint layers* - Ensure each city is visited exactly once
        
        **🎯 Step 4:** *Solution Extraction* - Get the optimal route from quantum state
        """)
        st.image("TensorNetworkTSP.png")
        st.markdown("""
        ## 🚀 How to Use This Interface

        **📊 Step 1: Input Your Problem**
        - 🏘️ **Number of Cities**: Choose your problem size (5-40 cities)
        - 🔗 **Max Connections**: For random problem generation, maximum number of cities that can be visited from each city
        - ✏️ **Manual Entry**: Edit the distance table directly  
        - 🎲 **Random Generator**: Create test problems instantly

        or

        - 📁 **Import Files**: Drag & drop CSV/JSON distance matrices
        
        **🎛️ Step 2: Configure Parameters** (Sidebar)
        - 🌡️ **Tau (τ)**: Imaginary time control (higher = shorter routes). If it is too large, the solver may have overflows. 
        - 🧱 **Layers**: More restriction layers = better constraints (but slower). If there are n-1 restriction layers, the solver is exact.
        
        
        **⚡ Step 3: Solve**
        - Hit the big 🚀 **Run Solver** button to use the solver
        
        **📈 Step 4: Analyze Results**
        - ⏱️ **Performance**: See how fast our algorithm is
        - 💰 **Cost**: Total distance of obtainer tour
        - 🗺️ **Route**: Step-by-step path visualization  
        - 📥 **Export**: Download results as JSON
        
        **💡 Pro Tip:** Start with small problems (5-7 cities) to understand the algorithm, then scale up!
        """)

# Enhanced Sidebar
st.sidebar.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; justify-content: center;'>
    <h3 style='color: white; margin: 0; font-size: 1.2rem;'>🎛️ Control Panel</h3>
</div>
""", unsafe_allow_html=True)


# Enhanced help text for file formats
with st.sidebar.expander("📋 📖 File Format Guide"):
    st.markdown("""
    📊 CSV Format Example:
    ```csv
    ,City 0,City 1,City 2,City 3,City 4
    City 0,0,10,15,20,25
    City 1,10,0,12,18,22
    City 2,15,12,0,14,16
    City 3,20,18,14,0,8
    City 4,25,22,16,8,0
    ```
    
    
    📝 JSON Format Examples:
    
    🔹 Simple:
    [[0,10,15],[10,0,20],[15,20,0]]
    
    🔹 Structured:
    {"distance_matrix": [[0,10,15],[10,0,20],[15,20,0]]}
    
    ✅ Requirements:
    
    🔸 Square matrix (same rows & columns)

    🔸 Numeric distances only
    
    🔸 Use 0 for diagonal elements
    """)

# File upload section with enhanced description
uploaded_file = st.sidebar.file_uploader(
    "🔼 Import Distance Matrix",
    type=["csv", "json"],
    help="🎯 Upload your own TSP problem! Supports CSV and JSON formats with automatic size detection."
)


if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        imported_matrix, imported_n_nodes = import_distance_matrix_from_csv(st, uploaded_file)
    elif file_type == 'json':
        imported_matrix, imported_n_nodes = import_distance_matrix_from_json(st, uploaded_file)
    else:
        st.sidebar.error("Unsupported file type")
        imported_matrix, imported_n_nodes = None, 0
    
    if imported_matrix is not None and imported_n_nodes > 0:
        st.sidebar.success(f"🎉 Import Successful!\n\n"
                          f"📊 Loaded {imported_n_nodes}×{imported_n_nodes} distance matrix\n\n"
                          f"🚀 Ready to solve your TSP problem!")
        
        # Update session state with imported data
        st.session_state.distance_matrix = imported_matrix
        st.session_state.imported_n_nodes = imported_n_nodes
        
        # Option to update n_nodes to match imported matrix
        if st.sidebar.button("🔄 ⚡ Auto-Resize to Match Import", use_container_width=True, help="Automatically adjust the number of cities to match your imported file"):
            st.session_state.auto_update_nodes = True
            st.rerun()

# Check if we need to auto-update n_nodes
if 'auto_update_nodes' in st.session_state and st.session_state.auto_update_nodes:
    if 'imported_n_nodes' in st.session_state:
        # This will be used to set the default value
        default_nodes = st.session_state.imported_n_nodes
        st.session_state.auto_update_nodes = False
    else:
        default_nodes = 7
else:
    default_nodes = 7

st.sidebar.markdown("---")

st.sidebar.markdown("<div align='center'>⚙️ Algorithm Configuration</div>", unsafe_allow_html=True)

# Tau parameter with enhanced description
tau = st.sidebar.slider(
    " Tau (τ) - Imaginary time evolution parameter", 
    min_value=0.1, 
    max_value=10.0, 
    value=1.0, 
    step=0.1,
    help="🔥 Controls optimization intensity! Higher values strongly favor shorter routes (risk: numerical overflow)."
)

# Number of nodes with enhanced description
n_nodes = st.sidebar.number_input(
    "🏘️ Number of Cities", 
    min_value=5, 
    max_value=40, 
    value=st.session_state.get('imported_n_nodes', default_nodes), 
    step=1,
    help="🎯 Choose your problem size! Start with 5-7 cities to learn, then scale up. Larger problems take more time."
)

# Maximum connections for random generation
max_connections = st.sidebar.number_input(
    "🔗 Max Connections (Random Mode)", 
    min_value=2, 
    max_value=n_nodes-1, 
    value=n_nodes-1, 
    step=1,
    help="🎲 For random problem generation: How connected should cities be? Full connectivity (n-1) = harder problems, partial = sparser graphs."
)

# Number of layers with enhanced description
n_layers = st.sidebar.number_input(
    "🧱 Tensor Network Restriction Layers", 
    min_value=1, 
    max_value=min(14,n_nodes-1), 
    value=min(14, n_nodes-1), 
    step=1,
    help="🏗️ More layers = stronger constraints = better solutions, but slower! Think of it as the 'depth' of quantum evolution. Max 14 to avoid memory issues."
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.5rem; border-radius: 6px; margin: 0.5rem 0;'>
    <h4 style='color: white; margin: 0; font-size: 0.95rem;'>💡 Tips & Troubleshooting</h4>
</div>

<div style='background: #1e1e1e; padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0; border-left: 2px solid #28a745;'>
    <p style='margin: 0; font-size: 0.8rem; color: #e0e0e0;'>
    <strong>🔧 Issues:</strong><br>
    🔸 Sequential result → Lower τ<br>
    🔸 Not optimal → Increase τ/layers<br>
    🔸 Repeated cities → More layers<br>
    🔸 Too slow → Reduce size/layers
    </p>
</div>

<div style='background: #1e1e1e; padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0; border-left: 2px solid #ff9800;'>
    <p style='margin: 0; font-size: 0.8rem; color: #e0e0e0;'>
    <strong>⚡ Performance:</strong><br>
    🔸 Start small (5-7 cities)<br>
    🔸 τ = 1.0 is good baseline<br>
    🔸 More layers = better quality<br>
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Main content area
st.header("📊 Problem Configuration")
# Initialize session state for distance matrix
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = torch.full((n_nodes, n_nodes), 0, dtype=torch.float)

# Update matrix size if n_nodes changed
if st.session_state.distance_matrix.size(0) != n_nodes:
    old_matrix = st.session_state.distance_matrix
    st.session_state.distance_matrix = torch.full((n_nodes, n_nodes), 0, dtype=torch.float)
    
    # Copy over existing values that fit
    min_size = min(old_matrix.shape[0], n_nodes)
    st.session_state.distance_matrix[:min_size, :min_size] = old_matrix[:min_size, :min_size]

col1, col2 = st.columns([3, 1])
with col2:
    st.subheader("⚡Quick Actions")
    
    # Generate random problem button with enhanced feedback
    if st.button("🎲 🌟 Generate Random Problem", type="primary", use_container_width=True, help="Create a random TSP instance with the current settings - perfect for testing!"):
        with st.spinner("🎨 Crafting your random TSP challenge..."):
            # Use the distance range from 1 to 10
            distance_range = 10.0
            random_distances = generate_problem(n_nodes, max_connections, distance_range)
            for i in range(random_distances.size(0)):
                for j in range(random_distances.size(1)):
                    if random_distances[i][j] == float('inf'):
                        random_distances[i][j] = 0
            st.session_state.distance_matrix = random_distances.float()
            st.success(f"🎉 Generated a {n_nodes}-city problem with max {max_connections} connections per city!")

    
    # Clear matrix button with confirmation
    if st.button("🗑️ 🧹 Clear All Distances", type="secondary", use_container_width=True, help="Reset the entire distance matrix to zeros"):
        st.session_state.distance_matrix = torch.full((n_nodes, n_nodes), 0, dtype=torch.float)
        st.info("🧽 Matrix cleared!")

    
    # Symmetrize matrix button with explanation
    if st.button("🔄 ⚖️ Make Symmetric", type="secondary", use_container_width=True, help="Ensure equal distances in both directions (A→B = B→A) for classic TSP"):
        matrix = st.session_state.distance_matrix
        # Make symmetric by taking minimum of (i,j) and (j,i)
        symmetric_matrix = torch.clone(matrix)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if symmetric_matrix[i,j] != 0 and symmetric_matrix[j,i] != 0:
                    symmetric_matrix[i,j] = min(symmetric_matrix[i,j], symmetric_matrix[j,i])
                elif symmetric_matrix[i,j] == 0 and symmetric_matrix[j,i] != 0:
                    symmetric_matrix[i,j] = symmetric_matrix[j,i]

        st.session_state.distance_matrix = symmetric_matrix
        st.info("🔄 Matrix symmetrized! Now A→B distance equals B→A distance.")
    # Export matrix as CSV button
    if st.button("📥 Export as CSV", type="secondary", use_container_width=True):
        # Convert the distance matrix to a DataFrame
        export_df = pd.DataFrame(
            st.session_state.distance_matrix.numpy(),
            columns=[f"City {i}" for i in range(n_nodes)],
            index=[f"City {i}" for i in range(n_nodes)]
        )
        
        # Convert DataFrame to CSV
        csv = export_df.to_csv()
        
        # Create download button
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="distance_matrix.csv",
            mime="text/csv",
            help="Download the distance matrix as a CSV file"
        )
st.markdown("---")

with col1:
    # Create two columns for the matrix editor section
    matrix_col1, matrix_col2 = st.columns(2)
    with matrix_col1:
        st.subheader("📋 Distance Matrix Editor")
        st.caption("Define distances between cities")
    with matrix_col2:
        st.markdown("""
        <div style='background: #1e1e1e; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 2px solid #ff9800;'>
            <p style='margin: 0; font-size: 0.85rem; color: #e0e0e0;'>
            🔸 <strong>0</strong> = No connection<br>
            🔸 <strong>Diagonal = 0</strong> (city to itself)<br>
            🔸 <strong>Click to edit</strong> distances directly
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize or update df_distances in session state
    st.session_state.df_distances = pd.DataFrame(
        st.session_state.distance_matrix.numpy(),
        columns=[f"🏢 To {i}" for i in range(n_nodes)],
        index=[f"🏢 From {i}" for i in range(n_nodes)]
    )
    
        
    # Create editable dataframe
    edited_df = st.data_editor(
        st.session_state.df_distances,
        use_container_width=True,
        num_rows="fixed",
        disabled=False,
        key="distance_editor"
    )

    # Convert back to tensor and update session state
    try:        
        # Update distance matrix tensor
        st.session_state.distance_matrix = torch.tensor(st.session_state.df_distances.values, dtype=torch.float32)
        
    except Exception as e:
        st.error(f"Error parsing distance matrix: {e}")




# Problem validation
matrix_sum = st.session_state.distance_matrix.sum().item()
has_connections = matrix_sum > 0

if has_connections:
    run_button = st.button(
        "🚀 Run Solver ✨", 
        type="primary", 
        use_container_width=True,
        help="Start the tensor network algorithm to find the optimal TSP solution!"
    )
else:
    st.markdown("""
    <div style='background: #fff3cd; padding: 0.6rem; border-radius: 6px; margin: 0.5rem 0; text-align: center;'>
        <p style='margin: 0; color: #856404; font-size: 0.95rem;'><strong>⚠️ Problem Not Ready</strong><br>
        📋 Please add distances first<br>
        🎲 Use "Generate Random" or import a file</p>
    </div>
    """, unsafe_allow_html=True)
    
    run_button = st.button(
        "🚀 Run Solver (Needs Data)", 
        type="secondary", 
        use_container_width=True,
        disabled=True,
        help="Add distances to your matrix first before solving!"
    )

if run_button and has_connections:
    with st.spinner("🌀 Tensor Network algorithm running... ⚛️"):
        # Run the solver with timing
        if st.session_state.distance_matrix.size(0) == n_nodes:
            st.session_state.distance_matrix_infs = torch.full((n_nodes, n_nodes), 0, dtype=torch.float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    st.session_state.distance_matrix_infs[i,j] = st.session_state.distance_matrix[i][j]
                    if st.session_state.distance_matrix[i][j] == 0:
                        st.session_state.distance_matrix_infs[i][j] = float('inf')
            try:
                start_time = perf_counter()
                solution = tn_tsp_solver(
                    distances= st.session_state.distance_matrix_infs,
                    tau=tau,
                    verbose=False,  # Disable verbose for cleaner output
                    n_layers=n_layers
                )
                
                end_time = perf_counter()
                runtime = end_time - start_time
                
                # Calculate solution cost
                solution_cost = cost_solution(solution, st.session_state.distance_matrix_infs)
                
                # Check for repeated cities
                has_repeated = has_repeated_elements(solution)
                
                # Store results in session state
                st.session_state.last_solution = solution
                st.session_state.last_cost = solution_cost
                st.session_state.last_runtime = runtime
                st.session_state.last_has_repeated = has_repeated
                
            except Exception as e:
                st.error(f"Error during solving: {str(e)}")
                st.stop()



st.markdown("---")

if 'last_solution' in st.session_state:
    if st.session_state.distance_matrix_infs.size(0) == n_nodes:
        # Enhanced Results Header
        st.markdown("""
        <div style='text-align: center; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 0rem; border-radius: 8px; margin: 1rem 0;'>
            <h3 style='color: white; margin: 0; font-size: 2rem;'>📊 Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Metrics row with better styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 0.3rem 0;'>
                <h4 style='margin: 0; font-size: 1.2rem;'>⏱️ Runtime</h4>
                <p style='margin: 0.1rem 0 0 0; font-size: 1.8rem; font-weight: bold;'>{st.session_state.last_runtime:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 0.3rem 0;'>
                <h4 style='margin: 0; font-size: 1.2rem;'>💰 Total Route Cost</h4>
                <p style='margin: 0.1rem 0 0 0; font-size: 1.8rem; font-weight: bold;'>{st.session_state.last_cost:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            quality_indicator = "✅ Perfect!" if not st.session_state.last_has_repeated else "⚠️ Has Repeats"
            quality_color = "#28a745" if not st.session_state.last_has_repeated else "#ffc107"
            quality_icon = "🎯" if not st.session_state.last_has_repeated else "⚠️"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {quality_color} 0%, {"#20c997" if not st.session_state.last_has_repeated else "#fd7e14"} 100%); color: white; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 0.3rem 0;'>
                <p style='margin: 0.1rem 0 0 0; font-size: 1.4rem; font-weight: bold;'>{quality_indicator}</p>
                <p style='margin: 0; font-size: 1rem; opacity: 0.8;'>{"Each city visited once" if not st.session_state.last_has_repeated else "Needs tuning"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced warning for repeated cities
        if st.session_state.last_has_repeated:
            st.markdown("""
            <div style='background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 0.6rem; border-radius: 6px; margin: 0.5rem 0;'>
                <h4 style='margin: 0; color: #856404; font-size: 1rem;'>🔧 Solution Needs Tuning</h4>
                <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem;'>
                Current solution visits some cities multiple times. Try:<br>
                🔸 <strong>Increase layers</strong> for stronger constraints<br>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Route display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("🗺️ Route Details")
            
            # Create enhanced route DataFrame
            route_data = []
            solution = st.session_state.last_solution
            total_distance = 0
            
            for i, city in enumerate(solution):
                if i < len(solution) - 1:
                    next_city = solution[i + 1]
                    distance = st.session_state.distance_matrix_infs[city, next_city].item()
                else:
                    # Return to start
                    next_city = solution[0]
                    distance = st.session_state.distance_matrix_infs[city, next_city].item()
                
                if distance != float('inf'):
                    total_distance += distance
                
                route_data.append({
                    "🚩 Step": i + 1,
                    "🏢 From": f"City {int(city.item())}",
                    "🎯 To": f"City {int(next_city)}",
                    "📏 Distance": f"{distance:.2f}" if distance != float('inf') else "∞"
                    })
                
            # Enhanced route summary
            route_sequence = " → ".join([str(int(city.item())) for city in solution])
            route_sequence += f" → {int(solution[0].item())}"  # Return to start
            
            st.code(route_sequence, language="text")

            route_df = pd.DataFrame(route_data)
            st.dataframe(route_df, use_container_width=True)
            
            # Generate JSON data
            json_data = export_solution_as_json(
                st.session_state.last_solution,
                st.session_state.last_cost,
                st.session_state.last_runtime,
                st.session_state.distance_matrix
            )
            
            # Enhanced download button
            st.download_button(
                label="📥 🚀 Download Complete Solution (JSON)",
                data=json_data,
                file_name=f"quantum_tsp_solution_{n_nodes}cities_{st.session_state.last_cost:.0f}dist.json",
                mime="application/json",
                help="📦 Download everything: route, cost, runtime, distance matrix, and metadata!",
                use_container_width=True
            )
            
            st.caption("💡 **JSON includes:** Route, cost, runtime, distance matrix, metadata & paper reference")
        
        with col2:
            st.subheader("🌐 Visualization")
            st.caption("Red arrows show optimal route, numbers in parentheses show visit order")
            
            # Create network graph
            try:
                G = nx.DiGraph()
                
                # Add nodes
                for i in range(n_nodes):
                    G.add_node(i)
                
                # Add edges from solution
                solution_list = [int(city.item()) for city in solution]
                
                # Add route edges
                for i in range(len(solution_list)):
                    current_city = solution_list[i]
                    next_city = solution_list[(i + 1) % len(solution_list)]  # Wrap around to start
                    distance = st.session_state.distance_matrix[current_city, next_city].item()
                    if distance != 0:
                        G.add_edge(current_city, next_city, weight=distance, route_order=i+1)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(8, 6))

                
                # Position nodes in a circle for better visualization
                pos = nx.circular_layout(G)
                
                # Draw all possible connections (faded)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j and st.session_state.distance_matrix[i, j] != 0:
                            nx.draw_networkx_edges(G, pos, [(i, j)], alpha=0.1, edge_color='gray', ax=ax)
                
                # Draw route edges (highlighted)
                route_edges = [(solution_list[i], solution_list[(i + 1) % len(solution_list)]) 
                               for i in range(len(solution_list))]
                # Draw nodes first
                # Create node colors - all lightblue except the first node in the route which is green
                node_colors = ['lightblue'] * n_nodes
                beginning = solution_list[0]
                node_colors[beginning] = 'green'
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, ax=ax, node_shape='s')
                
                    
                # Add node labels with route order
                node_labels = {solution_list[0]: f"Start\ncity={solution_list[0]}"}
                for i, city in enumerate(solution_list[1:]):
                    node_labels[city] = f"city={city}\n(t={i+2})"
                    
                # Create a dictionary for font colors - only first node in white
                font_colors = {}
                for node in G.nodes():
                    font_colors[node] = 'white' if node == solution_list[0] else 'black'
                    
                nx.draw_networkx_labels(G, pos, node_labels, font_size=10, ax=ax, font_color=font_colors)
                
                nx.draw_networkx_edges(G, pos, route_edges, edge_color='red', width=3, alpha=0.8, ax=ax, arrows=True, arrowsize=20, node_size=1500)
                    
                    
                    
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Try using a smaller problem size or ensure the distance matrix has valid connections.")

