import torch
import pandas as pd
import numpy as np

def load_coordinate_table_from_csv(csv_path: str, device: torch.device, 
                                  actual_vocab_size: int = None) -> torch.Tensor:
    """
    Load POI coordinate table from CSV file for geographic distance calculation.
    Handles vocabulary with special tokens by padding with zeros.
    
    Args:
        csv_path: Path to CSV file with columns [poi_token_id, lat, lon]
        device: Target device for the tensor
        actual_vocab_size: Total vocabulary size including special tokens
    
    Returns:
        torch.Tensor: (actual_vocab_size, 2) coordinate table [lat, lon]
    """
    try:
        df = pd.read_csv(csv_path)
        
        # CSV has columns: poi_token_id, lat, lon
        max_csv_token_id = int(df['poi_token_id'].max())
        csv_vocab_size = max_csv_token_id + 1
        
        # Use actual vocabulary size if provided, otherwise use CSV size
        final_vocab_size = actual_vocab_size if actual_vocab_size is not None else csv_vocab_size
        
        # Initialize coordinate table with zeros (special tokens will have zero coordinates)
        coordinate_table = torch.zeros((final_vocab_size, 2), device=device, dtype=torch.float32)
        
        # Fill in coordinates from CSV
        for _, row in df.iterrows():
            token_id = int(row['poi_token_id'])
            if token_id < final_vocab_size:  # Safety check
                lat = float(row['lat'])
                lon = float(row['lon'])
                
                # Check for NaN/Inf coordinates
                if np.isnan(lat) or np.isinf(lat) or np.isnan(lon) or np.isinf(lon):
                    print(f"Warning: Invalid coordinates for token {token_id}: lat={lat}, lon={lon}")
                    # Use zero coordinates for invalid entries
                    lat, lon = 0.0, 0.0
                
                coordinate_table[token_id] = torch.tensor([lat, lon], dtype=torch.float32)
        
        # Report statistics
        num_special_tokens = final_vocab_size - csv_vocab_size
        print(f"Loaded coordinate table from {csv_path}:")
        print(f"  CSV POI tokens: {csv_vocab_size}")
        print(f"  Special tokens (zero coords): {num_special_tokens}")
        print(f"  Total vocabulary size: {final_vocab_size}")
        
        return coordinate_table
        
    except Exception as e:
        print(f"Error loading coordinate table from {csv_path}: {e}")
        print("Length loss will be disabled")
        return None

def euclidean_distance_batch(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distances between consecutive coordinate pairs.
    
    Args:
        coords1: (B, L-1, 2) coordinates [lat, lon] in degrees
        coords2: (B, L-1, 2) coordinates [lat, lon] in degrees
    
    Returns:
        torch.Tensor: (B, L-1) distances in coordinate degrees (0.001-2.0 typical range)
    """
    # Simple Euclidean distance: sqrt((lat2-lat1)^2 + (lon2-lon1)^2)
    diff = coords2 - coords1  # (B, L-1, 2)
    distances = torch.norm(diff, p=2, dim=-1)  # (B, L-1)
    
    # Clamp to prevent extreme values (reasonable coordinate range)
    distances = torch.clamp(distances, min=0.0, max=10.0)
    
    # Check for NaN/Inf (should be very rare with Euclidean)
    distances = torch.where(torch.isnan(distances) | torch.isinf(distances), 
                          torch.zeros_like(distances), distances)
    
    return distances



def convert_to_sparse_adjacency(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert dense adjacency matrix to sparse format to save memory.
    
    Args:
        adjacency_matrix: Dense adjacency matrix (V, V)
    
    Returns:
        torch.Tensor: Sparse adjacency matrix
    """
    if adjacency_matrix.is_sparse:
        return adjacency_matrix  # Already sparse
    
    # Convert to sparse format
    if adjacency_matrix.dtype == torch.bool:
        # For boolean matrices
        indices = adjacency_matrix.nonzero().t()
        values = torch.ones(indices.size(1), dtype=torch.bool)
    else:
        # For float matrices
        indices = adjacency_matrix.nonzero().t()
        values = adjacency_matrix[indices[0], indices[1]]
    
    sparse_adj = torch.sparse_coo_tensor(indices, values, adjacency_matrix.shape)
    return sparse_adj.coalesce()

def sparse_indexing(adjacency_map: torch.Tensor, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    """
    Efficient sparse matrix indexing using linear index hashing.
    
    Args:
        adjacency_map: Sparse adjacency matrix (V, V)
        src_ids: Source indices (B, L-1)
        tgt_ids: Target indices (B, L-1)
    
    Returns:
        torch.Tensor: Adjacency weights (B, L-1)
    """
    # Get sparse matrix components
    sparse_indices = adjacency_map._indices()  # [2, nnz]
    sparse_values = adjacency_map._values()    # [nnz]
    vocab_size = adjacency_map.size(1)
    
    # Convert sparse matrix (row, col) pairs to linear indices
    sparse_linear_indices = sparse_indices[0] * vocab_size + sparse_indices[1]  # [nnz]
    
    # Convert query (src, tgt) pairs to linear indices
    src_flat = src_ids.flatten()
    tgt_flat = tgt_ids.flatten()
    query_linear_indices = src_flat * vocab_size + tgt_flat  # [B*(L-1)]
    
    # Sort sparse indices for efficient searching
    sorted_sparse_indices, sort_perm = torch.sort(sparse_linear_indices)
    sorted_sparse_values = sparse_values[sort_perm]
    
    # Use searchsorted to find positions
    positions = torch.searchsorted(sorted_sparse_indices, query_linear_indices)
    
    # Handle out-of-bounds positions
    positions = torch.clamp(positions, 0, len(sorted_sparse_indices) - 1)
    
    # Check if the found positions actually match our queries
    found_indices = sorted_sparse_indices[positions]
    matches = (found_indices == query_linear_indices)
    
    # Get values where matches exist, 0 otherwise
    result = torch.where(matches, sorted_sparse_values[positions], 
                        torch.zeros_like(query_linear_indices, dtype=sparse_values.dtype))
    
    # Reshape back to original shape
    return result.view(src_ids.shape)

def load_adjacency_matrix_sparse(path: str) -> torch.Tensor:
    """
    Load adjacency matrix and convert to sparse format for memory efficiency.
    
    Args:
        path: Path to the adjacency matrix file
    
    Returns:
        torch.Tensor: Sparse adjacency matrix
    """
    print(f"Loading adjacency matrix from {path}...")
    adjacency_matrix = torch.load(path)
    print(f"Original adjacency matrix shape: {adjacency_matrix.shape}")
    
    # Calculate memory usage
    if adjacency_matrix.is_sparse:
        original_memory = adjacency_matrix._nnz() * 8  # 8 bytes per element
        dense_memory = adjacency_matrix.shape[0] * adjacency_matrix.shape[1] * 8
        sparsity = (1 - adjacency_matrix._nnz() / (adjacency_matrix.shape[0] * adjacency_matrix.shape[1])) * 100
        print(f"Already sparse matrix: {adjacency_matrix._nnz()} non-zero elements")
        print(f"Memory usage: {original_memory / 1e9:.2f} GB (dense would be {dense_memory / 1e9:.2f} GB)")
        print(f"Sparsity: {sparsity:.2f}%")
    else:
        # Convert to sparse
        original_memory = adjacency_matrix.numel() * adjacency_matrix.element_size()
        print(f"Dense matrix: {adjacency_matrix.numel()} elements")
        print(f"Original memory usage: {original_memory / 1e9:.2f} GB")
        
        adjacency_matrix = convert_to_sparse_adjacency(adjacency_matrix)
        
        sparse_memory = adjacency_matrix._nnz() * 8  # 8 bytes per element
        sparsity = (1 - adjacency_matrix._nnz() / (adjacency_matrix.shape[0] * adjacency_matrix.shape[1])) * 100
        print(f"Converted to sparse: {adjacency_matrix._nnz()} non-zero elements")
        print(f"New memory usage: {sparse_memory / 1e9:.2f} GB")
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Memory reduction: {original_memory / sparse_memory:.1f}x")
    
    return adjacency_matrix
