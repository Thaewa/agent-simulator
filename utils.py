import numpy as np
def gaussian_attraction(x,y,x0,y0,spread,peak):
    return np.exp(-peak*((x-x0)**2 + (y-y0)**2)/spread)

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
