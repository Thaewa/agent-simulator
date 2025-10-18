import numpy as np
import networkx as nx
def gaussian_attraction(x,y,x0,y0,spread,peak):
    """
    Evaluates a 2D Gaussian function at positions (x,y) with
    mean (x0,y0) and spread. The function returns the value
    of the Gaussian at the given positions, scaled by the
    factor 'peak'.
    """
    return peak*np.exp(-((x-x0)**2 + (y-y0)**2)/spread)

def estimate_gradient(points, z, edge_order=2):
    """
    points: (N,2) array with columns (x,y) from a full rectangular grid (flattened).
    z:      (N,) array of scalar values aligned with 'points' rows.
    Returns:
        dzdx_flat, dzdy_flat of shape (N,), aligned with the input row order.
    """
    x = points[:,0].astype(float)
    y = points[:,1].astype(float)
    z = np.asarray(z, dtype=float).ravel()
    N = x.size

    # recover grid axes and indices
    xs, inv_x = np.unique(x, return_inverse=True)
    ys, inv_y = np.unique(y, return_inverse=True)
    nx, ny = xs.size, ys.size
    if nx * ny != N:
        raise ValueError("Input does not cover a full rectangular grid.")

    # check duplicates/missing points
    counting = np.zeros((ny, nx), dtype=int)
    counting[inv_y, inv_x] += 1
    if not np.all(counting == 1):
        raise ValueError("Grid has duplicates or missing points.")

    # place z onto grid (rows=y, cols=x), then differentiate
    Z = np.empty((ny, nx), float)
    Z[inv_y, inv_x] = z
    dZdy, dZdx = np.gradient(Z, ys, xs, edge_order=edge_order)  # note: returns (∂/∂y, ∂/∂x)

    # map back to original flat ordering
    return dZdx[inv_y, inv_x], dZdy[inv_y, inv_x]

def m_closest_rows(points: np.ndarray, location: np.ndarray, m: int):
    """
    points: (n, 2) array
    location: (2,) or (1,2)
    m: number of nearest rows

    Returns:
        idx_sorted: (m,) indices of nearest rows (ascending by distance)
        pts_sorted: (m, 2) rows at those indices
        d_sorted:   (m,) distances (included if return_distances=True)
    """
    points = np.asarray(points, dtype=float)
    location = np.asarray(location, dtype=float).reshape(2,)
    n = points.shape[0]
    if points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2)")
    if not (1 <= m <= n):
        raise ValueError("m must be between 1 and n")

    d = np.linalg.norm(points - location, axis=1)        # all distances
    idx_part = np.argpartition(d, m-1)[:m]               # top-m (unordered)
    order_local = np.argsort(d[idx_part], kind="stable") # order those m
    idx_sorted = idx_part[order_local]

    pts_sorted = points[idx_sorted]
    return idx_sorted, pts_sorted

# def grid_graph_from_array(arr,X,Y, connectivity=8):
#     """
#     Create a NetworkX graph from a 2D array where truthy cells are nodes.
#     Edges connect unit-neighbors (connectivity=4) or king-move neighbors (connectivity=8).
    
#     Node labels are (row, col) index tuples matching NumPy indexing.
#     """
#     if arr.ndim != 2:
#         raise ValueError("arr must be 2D")
#     G = nx.Graph()
    
#     for row in range(X.shape[0]):
#         for col in range(X.shape[1]):
#             if arr[row,col]:
#                 G.add_node((X[row,col],Y[row,col]))  
            
#     # Neighbor offsets
#     nbrs_4 = [(-1,0), (1,0), (0,-1), (0,1)]
#     nbrs_8 = nbrs_4 + [(-1,-1), (-1,1), (1,-1), (1,1)]
#     offsets = nbrs_4 if connectivity == 4 else nbrs_8
    
#     nodes = list(G.nodes())
    
#     for node in nodes:
#         for offset in offsets:
#             nbr = (node[0] + offset[0], node[1] + offset[1])
#             if nbr in nodes:
#                 G.add_edge(node, nbr)
#     return G


def grid_graph_from_array(arr, X, Y, connectivity=8):
    """
    Nodes are labeled by their coordinate values: (X[row,col], Y[row,col]).
    Edges connect 4- or 8-neighbors among cells where arr==True.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    H, W = arr.shape
    G = nx.Graph()
    # --- nodes: keep your coordinate-based labels ---
    for r in range(H):
        for c in range(W):
            G.add_node((X[r, c], Y[r, c]))

    # --- neighbor offsets ---
    nbrs_4 = [(-1,0), (1,0), (0,-1), (0,1)]
    nbrs_8 = nbrs_4 + [(-1,-1), (-1,1), (1,-1), (1,1)]
    offsets = nbrs_4 if connectivity == 4 else nbrs_8

    # --- edges: iterate in index space, map to coordinate labels ---
    for r in range(H):
        for c in range(W):
            if not arr[r, c]:
                continue
            u = (X[r, c], Y[r, c])
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and arr[nr, nc]:
                    v = (X[nr, nc], Y[nr, nc])
                    # optional duplicate guard (undirected graph)
                    if (nr > r) or (nr == r and nc > c):
                        G.add_edge(u, v)

    return G