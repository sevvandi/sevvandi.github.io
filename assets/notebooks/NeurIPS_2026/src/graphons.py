import numpy as np
from scipy.special import comb
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def ring_graphon(n, alpha):
    """
    Generate a ring graphon on an n x n grid.

    Parameters:
        n     : grid resolution
        alpha : bandwidth parameter

    Returns:
        n x n numpy array
    """
    x = np.arange(1, n + 1) / n  # (n,)
    y = np.arange(1, n + 1) / n  # (n,)

    X, Y = np.meshgrid(x, y, indexing='ij')  # both (n, n)

    mat = (
        0.9 * np.exp((-Y**2 - (X - 1)**2) / alpha**2) +
        0.9 * np.exp((-(Y - 1)**2 - X**2) / alpha**2) +
        0.9 * np.exp(-((np.sin(0.75 * np.pi) * X + np.cos(0.75 * np.pi) * Y) / alpha)**2)
    )

    return mat


def plot_graphon(W, cols=("white", "black")):
    """
    Plot a graphon/adjacency matrix as a heatmap.

    Parameters:
        W    : 2D numpy array
        cols : tuple of two colors for the gradient (default: white to black)
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cols)

    fig, ax = plt.subplots()

    ax.imshow(
        W,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[-0.5, n - 0.5, n - 0.5, -0.5]
    )

    # Border rectangle matching geom_rect in ggplot version
    rect = plt.Rectangle(
        (-0.5, -0.5), n, n,
        linewidth=0.5,
        edgecolor="black",
        facecolor="none"
    )
    ax.add_patch(rect)

    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def sbm_graphon(mat, n):
    k = mat.shape[0]
    block_size = n / k
    graphon = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            row_start = int(i * block_size)
            row_end = int((i + 1) * block_size)
            col_start = int(j * block_size)
            col_end = int((j + 1) * block_size)
            graphon[row_start:row_end, col_start:col_end] = mat[i, j]
    return graphon



def sample_graph_from_graphon(graphon, n):
    """
    Generate a graph with n nodes from a graphon (m x m matrix)
    
    Parameters:
    -----------
    graphon : numpy array of shape (m, m)
        The graphon matrix with values in [0, 1]
    n : int
        Number of nodes in the generated graph
    
    Returns:
    --------
    adj_matrix : numpy array of shape (n, n)
        Adjacency matrix of the generated graph
    """
    m = graphon.shape[0]
    
    # Sample n uniform points in [0, 1]
    u = np.random.uniform(0, 1, n)
    u = np.sort(u)  # Sort to maintain order for graphon indexing
    
    # Map each point to graphon indices
    indices = (u * m).astype(int)
    indices = np.clip(indices, 0, m-1)  # Ensure within bounds
    
    # Initialize adjacency matrix
    adj_matrix = np.zeros((n, n))
    
    # Generate edges based on graphon probabilities
    for i in range(n):
        for j in range(i+1, n):
            # Get probability from graphon
            prob = graphon[indices[i], indices[j]]
            
            # Generate edge with that probability
            if np.random.rand() < prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Undirected graph
    
    return adj_matrix


def scale_graphon(f_mat: np.ndarray) -> np.ndarray:
    """Scale f_mat from [0, 1] to [-1, 1]."""
    return 2 * f_mat - 1


def scale_back(f_mat: np.ndarray) -> np.ndarray:
    """Scale f_mat from [-1, 1] back to [0, 1]."""
    f_int = (f_mat + 1) / 2
    if np.any(f_int < 0) or np.any(f_int > 1):
        import warnings
        warnings.warn("Reconstructed graphon has values outside [0, 1]; clipping.")
        f_int = np.clip(f_int, 0, 1)
    return f_int




def arrange_matrix(mat):
    """
    Sort rows and columns by decreasing sum.

    Parameters:
        mat : 2D numpy array

    Returns:
        reordered 2D numpy array
    """
    mat = np.array(mat, dtype=float)

    row_order = np.argsort(mat.sum(axis=1))[::-1]
    mat = mat[row_order, :]

    col_order = np.argsort(mat.sum(axis=0))[::-1]
    mat = mat[:, col_order]

    return mat



def resize_graphon(W, n):
    """
    Resample a graphon matrix onto an n x n grid using nearest-neighbor
    interpolation, so graphons defined on different resolutions can be
    combined on a common grid.

    Parameters:
        W : 2D numpy array (m x m)
        n : target grid resolution

    Returns:
        n x n numpy array
    """
    W = np.array(W, dtype=float)
    m = W.shape[0]
    idx = np.clip((np.arange(n) * m / n).astype(int), 0, m - 1)
    return W[np.ix_(idx, idx)]


def graphon_addition(W1, W2, alpha=0.5, n=None):
    """
    Simple (convex) graphon addition: alpha * W1 + (1 - alpha) * W2.

    W1 and W2 are combined pointwise on a common grid, i.e. the two
    graphons are overlaid rather than kept as separate parts. This is
    the "W1 + W2" style combination (cf. 0.1*W1 + 0.9*W2 on the slides).

    Parameters:
        W1, W2 : 2D numpy arrays, values in [0, 1]. Need not be the
                 same resolution; both are resampled to a common grid.
        alpha  : weight on W1, in [0, 1]. W2 gets weight (1 - alpha).
        n      : resolution of the combined grid. Defaults to the
                 larger of W1's and W2's resolutions.

    Returns:
        n x n numpy array
    """
    if n is None:
        n = max(W1.shape[0], W2.shape[0])
    W1r = resize_graphon(W1, n)
    W2r = resize_graphon(W2, n)
    return alpha * W1r + (1 - alpha) * W2r


def _block_sizes(n, alpha):
    """Split n into two block sizes (n1, n2) with n1/n ~= alpha."""
    n1 = int(round(alpha * n))
    n1 = min(max(n1, 1), n - 1) if n > 1 else n
    n2 = n - n1
    return n1, n2


def disjoint_graphon_addition(W1, W2, alpha=0.5, n=None):
    """
    Disjoint-union graphon addition.

    Places W1 and W2 as two separate blocks along the diagonal of a
    single graphon, with block sizes proportional to alpha and
    (1 - alpha). The off-diagonal blocks are 0, so there are no edges
    between the two parts -- graphs sampled from the result look like
    two disconnected components (a "smiley face" made of two graphons
    stitched together, as on the slides).

    Parameters:
        W1, W2 : 2D numpy arrays, values in [0, 1].
        alpha  : fraction of the combined grid allocated to W1.
        n      : resolution of the combined grid. Defaults to
                 W1.shape[0] + W2.shape[0].

    Returns:
        n x n numpy array
    """
    if n is None:
        n = W1.shape[0] + W2.shape[0]
    n1, n2 = _block_sizes(n, alpha)

    W = np.zeros((n, n))
    W[:n1, :n1] = resize_graphon(W1, n1)
    W[n1:, n1:] = resize_graphon(W2, n2)
    return W


def disjoint_graphon_addition_with_join(W1, W2, W3, alpha, n=100):
    """
    Disjoint-union graphon addition with a join.
 
    Same block-diagonal layout as disjoint_graphon_addition, but the
    off-diagonal blocks (which connect the two parts) are filled with
    a third graphon W3 instead of 0 -- resized to the n1 x n2 join
    block, with its transpose used for the symmetric block below the
    diagonal. This mirrors the R implementation:
 
        join_disjoint_graphons(W1, W2, W3, alpha)
 
    Passing a constant graphon for W3 (e.g. np.full((k, k), p)) recovers
    a uniform-probability join; p=1 recovers the classic graph "join"
    (every vertex in part 1 connects to every vertex in part 2); p=0
    recovers disjoint_graphon_addition.
 
    Parameters:
        W1, W2 : 2D numpy arrays, values in [0, 1] -- the two diagonal
                 blocks.
        W3     : 2D numpy array, values in [0, 1] -- the graphon used
                 to fill the off-diagonal (join) blocks.
        alpha  : fraction of the combined grid allocated to W1, in
                 [0, 1]. n1 = floor(alpha * n), n2 = n - n1.
        n      : resolution of the combined grid (default: 100, as in
                 the R version).
 
    Returns:
        n x n numpy array
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
 
    n1 = int(np.floor(alpha * n))
    n2 = n - n1
 
    W1r = resize_graphon(W1, n1)
    W2r = resize_graphon(W2, n2)
    W3r = resize_graphon_rect(W3, n1, n2)
 
    W = np.zeros((n, n))
    W[:n1, :n1] = W1r
    W[n1:, n1:] = W2r
    W[:n1, n1:] = W3r
    W[n1:, :n1] = W3r.T
    return W

def resize_graphon_rect(W, n1, n2):
    """
    Resample a graphon matrix onto an n1 x n2 grid using nearest-neighbor
    interpolation (rectangular version of resize_graphon, used for the
    off-diagonal / join block, which need not be square).
 
    Parameters:
        W       : 2D numpy array (m x m)
        n1, n2  : target number of rows, columns
 
    Returns:
        n1 x n2 numpy array
    """
    W = np.array(W, dtype=float)
    m = W.shape[0]
    row_idx = np.clip((np.arange(n1) * m / n1).astype(int), 0, m - 1)
    col_idx = np.clip((np.arange(n2) * m / n2).astype(int), 0, m - 1)
    return W[np.ix_(row_idx, col_idx)]




def edge_density(A):
    n = A.shape[0]
    num_edges = A.sum() / 2
    return 2 * num_edges / (n * (n - 1))


def triangle_density(A):
    n = A.shape[0]
    num_triangles = np.trace(A @ A @ A) / 6
    return 6 * num_triangles / (n * (n - 1) * (n - 2))


def graphon_one_to_N(item=1, n=100, alpha=0.05):
    """
    Generate the graphons from the paper.
    item: integer specifying which graphon to generate
    n: number of grid points in each dimension (total n^2 points)
    alpha: parameter for the ring graphon (only used if item=14)
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if item == 1:
        W = X * Y
    elif item == 2:
        W = np.exp(-(X**0.7 + Y**0.7))
    elif item == 3:
        W = 0.25 * (X**2 + Y**2 + np.sqrt(X) + np.sqrt(Y))
    elif item == 4:
        W = (X + Y) / 2
    elif item == 5:
        W = 1 / (1 + np.exp(-2 * (X**2 + Y**2)))
    elif item == 6:
        W = 1 / (1 + np.exp(-(np.maximum(X, Y)**2 + np.minimum(X, Y)**2)))
    elif item == 7:
        W = np.exp(-np.maximum(X, Y)**0.75)
    elif item == 8:
        W = np.exp(-0.5 * (np.minimum(X, Y) + np.sqrt(X) + np.sqrt(Y)))
    elif item == 9:
        W = np.log(1 + np.maximum(X, Y))
    elif item == 10:
        W = np.abs(X - Y)
    elif item == 11:
        W = 1 - np.abs(X - Y)
    elif item == 12:
        mat = np.array([[0.8, 0], [0, 0]])
        W = sbm_graphon(mat, n)
    elif item == 13:
        mat = np.array([[0, 0.8], [0.8, 0]])
        W = sbm_graphon(mat, n)
    elif item == 14:
        W = ring_graphon(n, alpha=alpha)
    else:
        raise ValueError("Invalid item number. Must be between 1 and 14.")

    return W