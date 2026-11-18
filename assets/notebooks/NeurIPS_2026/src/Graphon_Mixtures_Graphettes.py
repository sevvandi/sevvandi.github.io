"""
Graphon_Mixtures_Graphettes.py

Python port of an R package for:
  - (U,W) mixture graphs: a dense graphon part (W) joined with a sparse,
    hub-heavy part built from a "line graphon" mass partition (U).
  - Graphettes: sparsified graphons optionally edited with extra star
    (hub) structure, rings, or turned into a spanning tree/forest.
  - Hub prediction: fitting a two-segment ("sparse" / "dense") piecewise
    log-log line to a graph's sorted degree sequence, and using it to
    extrapolate hub degrees to a bigger graph of the same type.

Ported from (uploaded) R source files:
    graphon_generation.R, graphettes.R, predict_hubs.R, utils.R,
    autoplot_extract_sparse.R

NOTE on `extract_sparse`: predict_hubs.R and autoplot_extract_sparse.R both
call a function `extract_sparse()` that was not among the uploaded R files
(it's presumably defined elsewhere in the original package). Its usage in
those two files makes its contract clear enough to reconstruct:
  - it sorts a graph's degrees in decreasing order, takes logs, and fits
    two linear segments (an initial steep "hub" segment and a flatter
    "bulk" segment) to the resulting curve,
  - it picks the breakpoint ("cutoff") that best separates the two
    segments (the classic "L-method" of Salvador & Chan, 2004, for
    elbow/knee detection),
  - `cutoff` doubles as `num_hubs`: the number of top-degree nodes
    considered to be hubs.
`extract_sparse` below reconstructs this using ordinary least squares for
each candidate breakpoint. If your original R package's `extract_sparse`
used a different fitting method, swap it in here -- `predict_hubs` and
`plot_extract_sparse` only rely on the returned dict's keys.

Random number generation:
Every function that involves randomness takes an optional `rng` argument
(a numpy `Generator`, e.g. `np.random.default_rng(0)`). If omitted, a
fresh `default_rng()` is created. Graph objects are `python-igraph`
Graph instances throughout, for consistency with earlier notebooks in
this project.
"""

import warnings

import numpy as np
import igraph as ig
from scipy.ndimage import zoom

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None


# =============================================================================
# utils.R
# =============================================================================

def arrange_matrix(mat):
    """
    Sort rows and columns of a matrix by decreasing sum.

    Parameters:
        mat : 2D array

    Returns:
        Reordered 2D numpy array
    """
    mat = np.asarray(mat, dtype=float)
    row_order = np.argsort(mat.sum(axis=1))[::-1]
    mat2 = mat[row_order, :]
    col_order = np.argsort(mat.sum(axis=0))[::-1]
    return mat2[:, col_order]


def plot_graphon(W, cols=("white", "black"), ax=None):
    """
    Plot a graphon/adjacency matrix as a heatmap.

    Parameters:
        W    : 2D array
        cols : tuple of two colors for the gradient (default: white to black)
        ax   : matplotlib axes (created if None)

    Returns:
        (fig, ax)
    """
    import matplotlib.colors as mcolors

    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cols)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(W, cmap=cmap, vmin=0, vmax=1, origin="upper", extent=[-0.5, n - 0.5, n - 0.5, -0.5])
    rect = plt.Rectangle((-0.5, -0.5), n, n, linewidth=0.5, edgecolor="black", facecolor="none")
    ax.add_patch(rect)
    ax.axis("off")
    return fig, ax


def create_exp_matrix(nrow, scalar):
    """
    Creates an nrow x nrow matrix where entry (i,j) = exp(-(i+j)/scalar),
    using 1-based (i,j) as in the original R version.

    Parameters:
        nrow   : dimension of the matrix
        scalar : the scalar in exp(-(i+j)/scalar)

    Returns:
        nrow x nrow numpy array
    """
    i = np.arange(1, nrow + 1).reshape(-1, 1)
    j = np.arange(1, nrow + 1).reshape(1, -1)
    return np.exp(-(i + j) / scalar)


def ring_graphon(n, alpha=0.1):
    """
    Creates a graphon supported near a diagonal "ring" (wrapping around).

    Parameters:
        n     : dimension of the matrix
        alpha : bandwidth parameter (default 0.1)

    Returns:
        n x n numpy array
    """
    x = np.arange(1, n + 1) / n
    y = np.arange(1, n + 1) / n
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (
        0.9 * np.exp((-Y ** 2 - (X - 1) ** 2) / alpha ** 2)
        + 0.9 * np.exp((-(Y - 1) ** 2 - X ** 2) / alpha ** 2)
        + 0.9 * np.exp(-((np.sin(0.75 * np.pi) * X + np.cos(0.75 * np.pi) * Y) / alpha) ** 2)
    )


def sbm_graphon(mat, n):
    """
    Creates a Stochastic Block Model graphon by expanding a k x k block
    probability matrix into an n x n piecewise-constant graphon.

    Parameters:
        mat : k x k matrix of block probabilities
        n   : dimension of the output matrix

    Returns:
        n x n numpy array
    """
    mat = np.asarray(mat, dtype=float)
    k = mat.shape[0]
    block_size = n / k
    graphon = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            row_start = int(np.floor(i * block_size))
            row_end = int(np.floor((i + 1) * block_size))
            col_start = int(np.floor(j * block_size))
            col_end = int(np.floor((j + 1) * block_size))
            graphon[row_start:row_end, col_start:col_end] = mat[i, j]
    return graphon


# =============================================================================
# graphon_generation.R
# =============================================================================

def scale_graphon(W, n):
    """
    Resize a graphon to an n x n matrix (bilinear resampling), suitable
    for sampling large adjacency matrices from a small graphon.

    Parameters:
        W : 2D array, values in [0,1]
        n : target dimension

    Returns:
        n x n numpy array, clipped to [0,1]
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    if m == n:
        return W.copy()
    out = zoom(W, n / m, order=1)
    # guard against off-by-one rounding in zoom's output shape
    if out.shape != (n, n):
        out2 = np.zeros((n, n))
        r = min(n, out.shape[0])
        c = min(n, out.shape[1])
        out2[:r, :c] = out[:r, :c]
        if r < n:
            out2[r:, :] = out2[r - 1, :]
        if c < n:
            out2[:, c:] = out2[:, c - 1:c]
        out = out2
    return np.clip(out, 0, 1)


def generate_adj_from_probs(probsmat, rng=None):
    """
    Samples a symmetric 0/1 adjacency matrix from a matrix of edge
    probabilities (upper triangle sampled independently, mirrored).

    Parameters:
        probsmat : n x n matrix of edge probabilities in [0,1]
        rng      : numpy Generator (optional)

    Returns:
        n x n numpy int array
    """
    rng = rng or np.random.default_rng()
    probsmat = np.asarray(probsmat, dtype=float)
    n = probsmat.shape[0]
    iu = np.triu_indices(n, k=1)
    probs = np.clip(probsmat[iu], 0, 1)
    vals = rng.binomial(1, probs)
    adj = np.zeros((n, n), dtype=int)
    adj[iu] = vals
    adj = adj + adj.T
    np.fill_diagonal(adj, 0)
    return adj


def sample_graphon(W, n, rng=None):
    """
    Samples a graph with n nodes from a (dense) graphon W.

    Parameters:
        W   : 2D array, values in [0,1]
        n   : number of nodes to sample
        rng : numpy Generator (optional)

    Returns:
        igraph.Graph (undirected)
    """
    rng = rng or np.random.default_rng()
    Wscaled = scale_graphon(W, n)
    adj = generate_adj_from_probs(Wscaled, rng=rng)
    return ig.Graph.Adjacency((adj > 0).tolist(), mode="undirected")


def empirical_graphon(gr, n=None):
    """
    Computes the empirical graphon (adjacency matrix, sorted by decreasing
    row/column sum) of a graph, optionally resized to n x n.

    Parameters:
        gr : igraph.Graph
        n  : optional target dimension

    Returns:
        2D numpy array
    """
    adj = np.array(gr.get_adjacency().data, dtype=float)
    adj2 = arrange_matrix(adj)
    if n is not None:
        adj2 = scale_graphon(adj2, n)
    return adj2


def generate_star_union(wts, n):
    """
    Generates a disjoint union of star graphs whose sizes are proportional
    to `wts` -- the "sparse" (hub) part of a (U,W) mixture graph.

    Parameters:
        wts : mass partition (proportions of the hub sizes); rescaled to
              sum to 1 if it doesn't already
        n   : (approximate) total number of nodes across all stars

    Returns:
        igraph.Graph (undirected)
    """
    wts = np.asarray(wts, dtype=float)
    if not np.isclose(wts.sum(), 1.0):
        wts = wts / wts.sum()
        warnings.warn("Rescaling weights to add to 1")

    n = int(n)
    star_sizes = np.maximum(np.ceil(n * wts).astype(int), 1)

    graphs = [ig.Graph.Star(int(s), mode="undirected") for s in star_sizes]
    gr = graphs[0]
    for g in graphs[1:]:
        gr = gr.disjoint_union(g)
    return gr


def graph_join(gr1, gr2, p=0.5, option=2, rng=None):
    """
    Joins two graphs. option=1 gives a plain disjoint union; option=2
    additionally adds `ceil(ecount(gr1) * p)` random edges between the
    two parts (drawing both endpoints uniformly over the joined vertex set).

    Parameters:
        gr1, gr2 : igraph.Graph objects
        p        : proportion of gr1's edge count to add as cross edges
                   (only used if option==2)
        option   : 1 (disjoint union) or 2 (disjoint union + random edges)
        rng      : numpy Generator (optional)

    Returns:
        igraph.Graph
    """
    rng = rng or np.random.default_rng()
    if not (isinstance(gr1, ig.Graph) and isinstance(gr2, ig.Graph)):
        raise TypeError("Both inputs must be igraph.Graph objects")

    n1, n2 = gr1.vcount(), gr2.vcount()
    joined = gr1.disjoint_union(gr2)

    if option == 2:
        edgecount = int(np.ceil(gr1.ecount() * p))
        node1 = rng.integers(0, n1 + n2, size=edgecount)
        node2 = rng.integers(0, n1 + n2, size=edgecount)
        joined.add_edges(list(zip(node1.tolist(), node2.tolist())))
        joined = joined.simplify()

    if "name" in gr1.vs.attributes() and "name" in gr2.vs.attributes():
        joined.vs["name"] = list(gr1.vs["name"]) + list(gr2.vs["name"])

    return joined


def sample_mixed_graph(W, wts, nd, ns, p=0.5, option=2, rng=None):
    """
    Samples a (U,W) mixture graph: a dense graphon part (W, nd nodes)
    joined with a sparse, star-structured part (mass partition wts,
    ns nodes) -- see `generate_star_union` for the sparse "U" part.

    Parameters:
        W      : dense graphon, symmetric matrix with values in [0,1]
        wts    : mass partition for the sparse/hub part (rescaled to sum
                 to 1 if needed)
        nd     : number of nodes in the dense part
        ns     : number of nodes in the sparse part
        p      : proportion of the dense part's edges added as cross
                 edges when joining (option==2 only)
        option : 1 (disjoint union) or 2 (disjoint union + random cross edges)
        rng    : numpy Generator (optional)

    Returns:
        igraph.Graph -- a sample from the (U,W) mixture
    """
    rng = rng or np.random.default_rng()
    W = np.asarray(W, dtype=float)
    if W.shape[0] != W.shape[1]:
        raise ValueError("W is not square")
    if not np.allclose(W, W.T):
        raise ValueError("W is not symmetric")
    if not (0 <= p <= 1):
        raise ValueError("p needs to be between 0 and 1")
    if option not in (1, 2):
        raise ValueError("option needs to be either 1 or 2")

    wts = np.asarray(wts, dtype=float)
    if not np.isclose(wts.sum(), 1.0):
        wts = wts / wts.sum()
        warnings.warn("Rescaling weights to add to 1")

    gr_dense = sample_graphon(W, int(nd), rng=rng)
    gr_sparse = generate_star_union(wts, int(ns))
    return graph_join(gr_dense, gr_sparse, p=p, option=option, rng=rng)


def line_graphon(probs):
    """
    Creates a "line graphon": a block-diagonal 0/1 matrix of disjoint
    cliques ("blocks"), with block sizes proportional to `probs`. This is
    the graphon U corresponding to a mass partition -- the object a (U,W)
    mixture's sparse/hub part is implicitly built from.

    Parameters:
        probs : list of probabilities (mass partition), largest first.
                If they sum to < 1, the remainder is treated as an
                (unfilled) leftover block.

    Returns:
        2D numpy array (0/1 block-diagonal matrix)
    """
    probs = np.asarray(probs, dtype=float)

    if np.isclose(probs.sum(), 1.0):
        use_probs = probs
        n_blocks = len(probs)
    else:
        zeroprob = 1 - probs.sum()
        use_probs = np.append(probs, zeroprob)
        n_blocks = len(probs)  # last (leftover) block is left unfilled, as in the original

    minp = use_probs.min()
    nn = int(3 * len(use_probs) / minp)
    if nn > 20000:
        warnings.warn(
            f"line_graphon: requested matrix size {nn}x{nn} is very large "
            f"(driven by a small weight, min={minp:.2g}) - capping at 20000x20000. "
            "Consider merging/dropping very small weights."
        )
        nn = 20000
    mat = np.zeros((nn, nn))

    st = 0
    for kk in range(n_blocks):
        if st < nn:
            prob_len = int(round(use_probs[kk] * nn))
            en = min(st + prob_len, nn)
            mat[st:en, st:en] = 1
            st = st + prob_len

    return mat


# =============================================================================
# graphettes.R
# =============================================================================

def sample_sparse_graphon(W, rho_n=None, n=100, rng=None):
    """
    Samples a graph from a sparsified graphon rho_n(n) * W, where rho_n is
    a (typically decreasing) sequence converging to 0.

    Parameters:
        W     : graphon, symmetric matrix with values in (0,1)
        rho_n : sparsity function of n (default: constant 1, i.e. dense).
                Must be non-increasing (rho_n(10) >= rho_n(100)).
        n     : number of nodes to sample
        rng   : numpy Generator (optional)

    Returns:
        igraph.Graph
    """
    rng = rng or np.random.default_rng()
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("Input W is not a matrix.")
    if W.shape[0] != W.shape[1]:
        raise ValueError("W is not square.")
    if not np.allclose(W, W.T):
        raise ValueError("W is not symmetric.")
    if W.max() >= 1 or W.min() <= 0:
        raise ValueError("W contains values outside of (0,1).")

    if rho_n is None:
        rho_n = lambda k: 1.0
    elif rho_n(10) < rho_n(100):
        raise ValueError("Function rho_n cannot be increasing!")

    Wn = rho_n(n) * W
    return sample_graphon(Wn, n, rng=rng)


def star_f1(xvals, t):
    """Star (hub) intensity function 1: exp(-(x-0.2)*t)."""
    xvals = np.asarray(xvals, dtype=float)
    return np.exp(-1 * (xvals - 0.2) * t)


def star_f2(xvals, t):
    """Star (hub) intensity function 2: t * exp(-(x-0.2))."""
    xvals = np.asarray(xvals, dtype=float)
    return t * np.exp(-1 * (xvals - 0.2))


def star_f3(xvals, t):
    """Star (hub) intensity function 3: t * exp(-(x+1))."""
    xvals = np.asarray(xvals, dtype=float)
    return t * np.exp(-1 * (xvals + 1))


def star_f4(xvals, t):
    """Star (hub) intensity function 4: constant, k*t with k=3."""
    xvals = np.asarray(xvals, dtype=float)
    k = 3
    return np.full(xvals.shape, k * t, dtype=float)


def star_f5(xvals, t):
    """Star (hub) intensity function 5: (t/10) * exp(-(x-0.2))."""
    xvals = np.asarray(xvals, dtype=float)
    return t / 10 * np.exp(-1 * (xvals - 0.2))


#: Registry of the named star functions, as used by sample_graphette's
#: graph_edit_f argument ('star_f1', ..., 'star_f5').
STAR_FUNCTIONS = {
    "star_f1": star_f1,
    "star_f2": star_f2,
    "star_f3": star_f3,
    "star_f4": star_f4,
    "star_f5": star_f5,
}


def sample_star_function(star_f, num_samples, t, rng=None):
    """
    Samples the number of star edges to add at each of `num_samples`
    (evenly spaced) graphon positions, as Poisson(star_f(x, t)).

    Parameters:
        star_f      : one of star_f1..star_f5 (or a custom callable(x, t))
        num_samples : number of positions (= number of nodes)
        t           : star-function parameter (hub strength)
        rng         : numpy Generator (optional)

    Returns:
        numpy int array of length num_samples
    """
    rng = rng or np.random.default_rng()
    xvals = np.linspace(0, 1, num_samples)
    star_vals = np.clip(star_f(xvals, t), 0, None)
    return rng.poisson(star_vals)


def add_star_edges(gr, star_f, t, rng=None):
    """
    Adds extra "star" (hub) structure to a graph: for each existing node
    i, draws a Poisson(star_f(x_i, t)) number of new leaf vertices and
    connects them to i.

    Parameters:
        gr     : igraph.Graph
        star_f : one of star_f1..star_f5 (or a custom callable(x, t))
        t      : star-function parameter (hub strength)
        rng    : numpy Generator (optional)

    Returns:
        igraph.Graph (modified copy)
    """
    rng = rng or np.random.default_rng()
    gr = gr.copy()
    num_nodes = gr.vcount()
    star_samples = sample_star_function(star_f, num_nodes, t, rng=rng)

    star_inds = np.where(star_samples > 0)[0]
    for i in star_inds:
        num_star_edges = int(star_samples[i])
        new_vertices = list(range(num_nodes, num_nodes + num_star_edges))
        gr.add_vertices(num_star_edges)
        gr.add_edges([(int(i), v) for v in new_vertices])
        num_nodes = gr.vcount()

    return gr


def add_rings(gr, proportion=0.1, ring_sizes=(5, 6), rng=None):
    """
    Adds small ring (cycle) subgraphs to a graph, each connected to the
    main graph by a single random edge.

    Parameters:
        gr         : igraph.Graph
        proportion : fraction of the graph's node count determining how
                     many rings to add (at least 1)
        ring_sizes : candidate ring sizes, sampled uniformly per ring
        rng        : numpy Generator (optional)

    Returns:
        igraph.Graph (modified copy)
    """
    rng = rng or np.random.default_rng()
    if not isinstance(gr, ig.Graph):
        raise TypeError("Input must be an igraph.Graph")
    if not (0 < proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1")

    n_vertices = gr.vcount()
    n_rings = max(1, round(n_vertices * proportion))

    result = gr.copy()
    for _ in range(n_rings):
        ring_size = int(rng.choice(ring_sizes))
        ring = ig.Graph.Ring(ring_size)
        current_n = result.vcount()
        result = result.disjoint_union(ring)

        selected_node = int(rng.integers(0, current_n))
        ring_node = int(rng.integers(current_n, current_n + ring_size))
        result.add_edges([(selected_node, ring_node)])

    return result


def remove_cycles(gr):
    """
    Removes cycles from a graph by taking a spanning tree (per connected
    component -- i.e. a spanning forest for disconnected graphs).

    Parameters:
        gr : igraph.Graph

    Returns:
        igraph.Graph (spanning tree/forest)
    """
    gr = gr.as_undirected() if gr.is_directed() else gr
    return gr.spanning_tree()


def sample_graphette(W, rho_n=None, graph_edit_f=None, n=100, t_or_p=0.5, ring_sizes=(5, 6), rng=None):
    """
    Samples a graph from a "graphette" (W, rho_n, f): a sparsified graphon
    rho_n(n)*W, optionally edited by f -- adding extra star/hub structure,
    adding rings, or removing cycles (spanning tree/forest).

    Parameters:
        W             : graphon, symmetric matrix with values in (0,1)
        rho_n         : sparsity function of n (default: constant 1)
        graph_edit_f  : None, one of 'star_f1'..'star_f5', 'add_rings', or
                        'remove_cycles'
        n             : number of nodes to sample
        t_or_p        : parameter for the star function (hub strength) or
                        the ring function (proportion, in [0,1])
        ring_sizes    : candidate ring sizes (only used for 'add_rings')
        rng           : numpy Generator (optional)

    Returns:
        igraph.Graph
    """
    rng = rng or np.random.default_rng()
    gr1 = sample_sparse_graphon(W, rho_n, n, rng=rng)

    if graph_edit_f is None:
        return gr1
    if graph_edit_f in STAR_FUNCTIONS:
        return add_star_edges(gr1, STAR_FUNCTIONS[graph_edit_f], t_or_p, rng=rng)
    if graph_edit_f == "add_rings":
        if not (0 <= t_or_p <= 1):
            raise ValueError("Parameter t_or_p needs to be between 0 and 1 for rings!")
        return add_rings(gr1, proportion=t_or_p, ring_sizes=ring_sizes, rng=rng)
    if graph_edit_f == "remove_cycles":
        return remove_cycles(gr1)

    raise ValueError(f"Unknown graph_edit_f: {graph_edit_f!r}")


# =============================================================================
# extract_sparse (reconstructed -- see module docstring) + predict_hubs.R
# + autoplot_extract_sparse.R
# =============================================================================

def fit_two_segments(y):
    """
    Fits two OLS lines to y (against x=1..len(y)), trying every possible
    changepoint i in [2, n-1] and keeping the one that minimizes the
    length-weighted average MSE across both segments:
        mse = (i*mse1 + (n-i)*mse2) / n

    Parameters:
        y : 1D array

    Returns:
        dict with keys:
            'num_hubs' / 'cutoff' : the best changepoint i (both equal;
                                     'num_hubs' may be overwritten by
                                     extract_sparse)
            'models'  : (coeffs1, coeffs2), each [slope, intercept]
                        from np.polyfit(x, y, 1), fit on segments
                        1..i and (i+1)..n respectively
            'mse'     : the best (lowest) weighted MSE found
            'data'    : dict with 'x' (1..n) and 'y' (the input y)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError(
            f"fit_two_segments needs at least 3 points to fit two segments (got {n}). "
            "In extract_sparse, this means the above-median filtering left too few "
            "distinct degree values -- the graph's degree sequence may be too small "
            "or too uniform for this hub-detection method."
        )
    x = np.arange(1, n + 1, dtype=float)

    best_mse, best_i, best_models = np.inf, None, None
    for i in range(2, n):  # matches R's 2:(n-1) inclusive
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]
        c1 = np.polyfit(x1, y1, 1)
        c2 = np.polyfit(x2, y2, 1)
        mse1 = np.mean((y1 - np.polyval(c1, x1)) ** 2)
        mse2 = np.mean((y2 - np.polyval(c2, x2)) ** 2)
        mse = (i * mse1 + (n - i) * mse2) / n
        if mse < best_mse:
            best_mse, best_i, best_models = mse, i, (c1, c2)

    return {
        "num_hubs": best_i,
        "cutoff": best_i,
        "models": best_models,
        "mse": best_mse,
        "data": {"x": x, "y": y},
    }


def extract_sparse(gr):
    """
    Estimates the hub / bulk split of a graph's degree sequence, and the
    corresponding mass-partition (sparse-component) probabilities.

    Steps (matching the R version exactly):
        1. degs = sorted, unique, decreasing degree values;
           degs_nonuniq = sorted, decreasing degrees (all nodes, with repeats).
        2. log_degs = log(degs + 1), then keep only values above their
           own median (this trims off the flat "bulk" tail before fitting).
        3. fit_two_segments(log_degs above median) -> a changepoint kk
           into that filtered (but still order-consistent, since it's a
           prefix of the full sorted-descending list) vector.
        4. Convert kk into an actual node count: threshold = degs[kk],
           kknew = number of (non-unique) node degrees >= threshold.
        5. phat = the top kknew node degrees, renormalized to sum to 1
           (the estimated mass partition / sparse-component probabilities).

    Parameters:
        gr : igraph.Graph

    Returns:
        dict (as returned by fit_two_segments, i.e. with 'data', 'cutoff',
        'models', 'mse' referring to the *filtered* log-degree fit) plus:
            'num_hubs' : overwritten with kknew -- the actual number of
                         top-degree nodes above the estimated cutoff
                         (NOT the same as 'cutoff', which still indexes
                         the filtered curve -- this mirrors the R code,
                         where only num_hubs is reassigned)
            'phat'     : estimated mass partition, length kknew
    """
    degs_nonuniq = np.array(sorted(gr.degree(), reverse=True), dtype=float)
    degs = np.array(sorted(np.unique(gr.degree()), reverse=True), dtype=float)

    log_degs = np.log(degs + 1)
    log_degs_filtered = log_degs[log_degs > np.median(log_degs)]

    out = fit_two_segments(log_degs_filtered)
    kk = out["num_hubs"]
    threshold = degs[kk - 1]  # degs is a prefix-consistent superset of log_degs_filtered
    kknew = int(np.sum(degs_nonuniq >= threshold))
    out["num_hubs"] = kknew

    degree_sum = degs_nonuniq[:kknew].sum()
    out["phat"] = degs_nonuniq[:kknew] / degree_sum

    return out


def predict_hubs(gr, n, k=None):
    """
    Predicts the degree of hub nodes in an unseen, larger (or smaller)
    graph of size n, assumed generated by the same process as `gr`, by
    linearly rescaling the current hub degrees by n / vcount(gr).

    Parameters:
        gr : the observed graph
        n  : number of nodes in the new (future) graph
        k  : number of hubs to predict; if None, taken from
             extract_sparse(gr)['num_hubs']

    Returns:
        numpy int array of predicted hub degrees, length k
    """
    if k is None:
        k = extract_sparse(gr)["num_hubs"]

    current_hub_degs = np.array(sorted(gr.degree(), reverse=True)[:k], dtype=float)
    num_nodes = gr.vcount()
    return np.round(current_hub_degs / num_nodes * n).astype(int)


def plot_extract_sparse(result, ax=None):
    """
    Plots the output of `extract_sparse`: the sorted log-degree sequence,
    with the two fitted ("Sparse" / "Dense") line segments overlaid and
    the breakpoint marked.

    Parameters:
        result : dict, the output of extract_sparse()
        ax     : matplotlib axes (created if None)

    Returns:
        ax
    """
    data = result["data"]
    x, y = data["x"], data["y"]
    n = len(x)
    i = result["cutoff"]
    c1, c2 = result["models"]

    x_range1 = np.linspace(1, i, 100)
    x_range2 = np.linspace(i + 1, n, 100)
    y_pred1 = np.polyval(c1, x_range1)
    y_pred2 = np.polyval(c2, x_range2)

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x, y, s=8, color="black", zorder=2)
    ax.plot(x_range1, y_pred1, color="tab:blue", linewidth=2, label="Sparse")
    ax.plot(x_range2, y_pred2, color="tab:orange", linewidth=2, label="Dense")
    ax.scatter([i], [y[i - 1]], color="red", zorder=5, label=f"Elbow (k={i})")
    ax.set_xlabel("Index")
    ax.set_ylabel("Log Degree")
    ax.legend()
    return ax
