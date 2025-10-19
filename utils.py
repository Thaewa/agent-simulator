from __future__ import annotations

import numpy as np
import networkx as nx


def gaussian_attraction(x, y, x0, y0, spread, peak):
    r"""
    Evaluate a 2D Gaussian-like attraction field.

    Computes
    :math:`f(x,y) = \text{peak} \cdot \exp\!\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{\text{spread}}\right)`.

    :param x: X-coordinates (array-like or scalar).
    :type x: numpy.ndarray | float | int
    :param y: Y-coordinates (array-like or scalar).
    :type y: numpy.ndarray | float | int
    :param x0: X-coordinate of the Gaussian center.
    :type x0: float | int
    :param y0: Y-coordinate of the Gaussian center.
    :type y0: float | int
    :param spread: Spread parameter in the exponent denominator (must be > 0).
    :type spread: float
    :param peak: Amplitude scaling factor.
    :type peak: float
    :return: Array (or scalar) of evaluated values, broadcast to the shape of ``x``/``y``.
    :rtype: numpy.ndarray | float
    """
    return peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / spread)


def estimate_gradient(points, z, edge_order: int = 2):
    r"""
    Estimate gradients :math:`\partial z/\partial x` and :math:`\partial z/\partial y`
    on a **full rectangular** grid provided in flattened form.

    :param points: ``(N, 2)`` array with columns ``(x, y)`` from a full rectangular grid (flattened).
    :type points: numpy.ndarray
    :param z: ``(N,)`` array of scalar field values aligned with the rows of ``points``.
    :type z: numpy.ndarray
    :param edge_order: Order of the edge scheme used by ``numpy.gradient`` (1 or 2).
    :type edge_order: int
    :return: Tuple ``(dzdx_flat, dzdy_flat)``, each of shape ``(N,)`` and aligned with input row order.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If ``points`` do not cover a full rectangular grid or contain duplicates.
    """
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    z = np.asarray(z, dtype=float).ravel()
    N = x.size

    # recover grid axes and indices
    xs, inv_x = np.unique(x, return_inverse=True)
    ys, inv_y = np.unique(y, return_inverse=True)
    nx_, ny_ = xs.size, ys.size
    if nx_ * ny_ != N:
        raise ValueError("Input does not cover a full rectangular grid.")

    # check duplicates/missing points
    counting = np.zeros((ny_, nx_), dtype=int)
    counting[inv_y, inv_x] += 1
    if not np.all(counting == 1):
        raise ValueError("Grid has duplicates or missing points.")

    # place z onto grid (rows=y, cols=x), then differentiate
    Z = np.empty((ny_, nx_), float)
    Z[inv_y, inv_x] = z
    dZdy, dZdx = np.gradient(Z, ys, xs, edge_order=edge_order)  # returns (∂/∂y, ∂/∂x)

    # map back to original flat ordering
    return dZdx[inv_y, inv_x], dZdy[inv_y, inv_x]


def m_closest_rows(points: np.ndarray, location: np.ndarray, m: int):
    r"""
    Select the ``m`` closest rows in ``points`` to a target ``location``.

    :param points: Array of shape ``(n, 2)`` with XY coordinates.
    :type points: numpy.ndarray
    :param location: Target position; shape ``(2,)`` or ``(1, 2)``.
    :type location: numpy.ndarray
    :param m: Number of nearest rows to return (``1 <= m <= n``).
    :type m: int
    :return: ``(idx_sorted, pts_sorted)`` where
             ``idx_sorted`` are the indices of the nearest rows (ascending by distance)
             and ``pts_sorted`` are the corresponding rows.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If ``points`` is not ``(n, 2)`` or ``m`` is out of range.
    """
    points = np.asarray(points, dtype=float)
    location = np.asarray(location, dtype=float).reshape(2,)
    n = points.shape[0]
    if points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2)")
    if not (1 <= m <= n):
        raise ValueError("m must be between 1 and n")

    d = np.linalg.norm(points - location, axis=1)        # all distances
    idx_part = np.argpartition(d, m - 1)[:m]             # top-m (unordered)
    order_local = np.argsort(d[idx_part], kind="stable") # order those m
    idx_sorted = idx_part[order_local]
    pts_sorted = points[idx_sorted]
    return idx_sorted, pts_sorted


def in_circle(x, y, radius):
    r"""
    Check whether coordinates fall inside or on a centered circle.

    Tests :math:`x^2 + y^2 \le \text{radius}^2` elementwise.

    :param x: X-coordinates.
    :type x: numpy.ndarray | float | int
    :param y: Y-coordinates.
    :type y: numpy.ndarray | float | int
    :param radius: Circle radius.
    :type radius: float | int
    :return: Boolean mask (broadcast to input shape) indicating membership.
    :rtype: numpy.ndarray | bool
    """
    return x**2 + y**2 <= radius**2


def not_in_circle_center(x, y, radius):
    r"""
    Check whether coordinates lie strictly off the centered circle boundary.

    Tests :math:`x^2 + y^2 \ne \text{radius}^2` elementwise.

    :param x: X-coordinates.
    :type x: numpy.ndarray | float | int
    :param y: Y-coordinates.
    :type y: numpy.ndarray | float | int
    :param radius: Circle radius.
    :type radius: float | int
    :return: Boolean mask (broadcast to input shape) indicating non-membership of the boundary.
    :rtype: numpy.ndarray | bool
    """
    return x**2 + y**2 != radius**2


def grid_graph_from_array(arr, X, Y, connectivity: int = 8):
    r"""
    Build a grid graph over active cells with node labels given by their coordinates.

    Nodes are labeled by coordinate values ``(X[row, col], Y[row, col])``.
    Edges connect 4- or 8-neighbors among cells where ``arr == True``.

    :param arr: Boolean mask of active cells; shape ``(H, W)``.
    :type arr: numpy.ndarray
    :param X: X-coordinate array matching ``arr``; shape ``(H, W)``.
    :type X: numpy.ndarray
    :param Y: Y-coordinate array matching ``arr``; shape ``(H, W)``.
    :type Y: numpy.ndarray
    :param connectivity: Neighborhood type: ``4`` (Von Neumann) or ``8`` (Moore). Defaults to ``8``.
    :type connectivity: int
    :return: Undirected graph with nodes labeled by coordinate tuples.
    :rtype: networkx.Graph
    :raises ValueError: If ``arr`` is not 2D.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    H, W = arr.shape
    G = nx.Graph()

    # nodes: coordinate-based labels
    for r in range(H):
        for c in range(W):
            G.add_node((X[r, c], Y[r, c]))

    # neighbor offsets
    nbrs_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nbrs_8 = nbrs_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = nbrs_4 if connectivity == 4 else nbrs_8

    # edges
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