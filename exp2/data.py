from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import cKDTree


@dataclass
class Dataset:
    x: np.ndarray
    coordinates: np.ndarray
    types: np.ndarray
    type_names: list
    tissue: np.ndarray
    perturbation: np.ndarray
    eligible: np.ndarray
    genes: list
    cell_ids: list


def load_data(path, coordinates=None, type_key="celltype2", perturbation_key="perturbation",
              tissue_key=None, control_label="Control", assume_single_tissue=False, transform="log1p"):
    import anndata
    from scipy import sparse
    a = anndata.read_h5ad(path)
    raw = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Expected finite, nonnegative expression in AnnData.X")
    if not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("Cell IDs and gene names must be unique")
    x = np.log1p(raw) if transform == "log1p" else raw.copy()
    coords = (np.loadtxt(coordinates, delimiter=",", dtype=np.float64) if coordinates
              else np.asarray(a.obsm["spatial"], dtype=np.float64))
    if coords.ndim != 2 or len(coords) != len(x) or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be finite and row-aligned with expression")
    if a.obs[type_key].isna().any():
        raise ValueError("Missing cell types; explicitly label unknown cell types before loading")
    type_names, types = np.unique(a.obs[type_key].astype(str), return_inverse=True)
    perturbation = np.asarray(a.obs[perturbation_key].astype(object).fillna("__unknown__"), dtype=str)
    if tissue_key:
        if a.obs[tissue_key].isna().any():
            raise ValueError("Missing tissue IDs")
        tissue = a.obs[tissue_key].astype(str).to_numpy()
    elif assume_single_tissue:
        tissue = np.full(len(x), "single_tissue_assumed")
    else:
        raise ValueError("Supply --tissue-key or explicitly acknowledge --assume-single-tissue")
    return Dataset(x, coords, types, type_names.tolist(), tissue, perturbation,
                   perturbation == control_label, a.var_names.tolist(), a.obs_names.tolist())


def split_receivers(data, fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for tissue in np.unique(data.tissue):
        for t in np.unique(data.types):
            ids = np.flatnonzero(data.eligible & (data.tissue == tissue) & (data.types == t))
            rng.shuffle(ids)
            n_val = max(1, int(round(len(ids) * fraction))) if len(ids) >= 2 else 0
            val.extend(ids[:n_val])
            train.extend(ids[n_val:])
    if len(train) < 3 or not val:
        raise ValueError("Need at least three training and one validation eligible receivers")
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class AllCellGraph:
    """Generate complete within-tissue rows lazily; never allocate an N x N matrix."""

    def __init__(self, data, power, epsilon):
        self.data, self.power, self.epsilon = data, power, epsilon
        self.pools = {t: np.flatnonzero(data.tissue == t) for t in np.unique(data.tissue)}

    def __len__(self):
        return len(self.data.x)

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError(i)
        src = self.pools[self.data.tissue[i]]
        src = src[src != i]
        distances = np.linalg.norm(self.data.coordinates[src] - self.data.coordinates[i], axis=1)
        logits = -self.power * np.log(distances + self.epsilon)
        weight = np.exp(logits - logits.max()) if len(logits) else logits
        if len(weight):
            weight /= weight.sum()
        return src, weight.astype(np.float32)


class NullGraph:
    """Lazy deterministic sender reassignment with original distance-weight slots."""

    def __init__(self, data, graph, seed, mapping, pools, rewire):
        self.data, self.graph, self.seed = data, graph, seed
        self.mapping, self.pools, self.rewire = mapping, pools, rewire

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, receiver):
        src, weight = self.graph[receiver]
        new_src = self.mapping[src].copy()
        if self.rewire:
            pool = self.pools[(self.data.tissue[receiver],)]
            pool = pool[pool != receiver]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, receiver]))
            new_src = rng.choice(pool, len(src), replace=False) if len(src) else src.copy()
        else:
            new_src[new_src == receiver] = src[new_src == receiver]
        return new_src, weight


def make_graph(data, neighbors=16, radius=None, power=1.0, epsilon=1e-3, mode="knn"):
    """Incoming k-nearest neighbors within each physical tissue, optionally radius-limited."""
    if mode == "all":
        if radius is not None:
            raise ValueError("all mode uses every same-tissue cell; radius is incompatible")
        return AllCellGraph(data, power, epsilon)
    if mode != "knn":
        raise ValueError("graph mode must be knn or all")
    graph = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)) for _ in data.x]
    for tissue in np.unique(data.tissue):
        ids = np.flatnonzero(data.tissue == tissue)
        tree = cKDTree(data.coordinates[ids])
        distances, positions = tree.query(data.coordinates[ids], k=min(neighbors + 1, len(ids)))
        if distances.ndim == 1:
            distances, positions = distances[:, None], positions[:, None]
        for local, i in enumerate(ids):
            src = ids[positions[local]]
            d = distances[local]
            keep = src != i
            if radius is not None:
                keep &= d <= radius
            src, d = src[keep][:neighbors], d[keep][:neighbors]
            weight = 1.0 / (d + epsilon) ** power
            if len(weight):
                weight /= weight.sum()
            graph[i] = (src, weight.astype(np.float32))
    return graph


def environment(data, graph, control_label):
    labels = np.unique(data.perturbation)
    q = np.zeros((len(data.x), len(data.type_names) + len(labels) + 3), dtype=np.float32)
    names = (["neighbor_type:" + t for t in data.type_names] +
             ["neighbor_perturbation:" + p for p in labels] +
             ["log1p_nearest_perturbed_distance", "has_perturbed_in_tissue", "has_neighbors"])
    for i, (src, weight) in enumerate(graph):
        if len(src):
            q[i, :len(data.type_names)] = np.bincount(data.types[src], weights=weight,
                                                     minlength=len(data.type_names))
            for j, label in enumerate(labels):
                q[i, len(data.type_names) + j] = weight[data.perturbation[src] == label].sum()
            q[i, -1] = 1
    for tissue in np.unique(data.tissue):
        ids = np.flatnonzero(data.tissue == tissue)
        pert = ids[(data.perturbation[ids] != control_label) &
                   (data.perturbation[ids] != "__unknown__")]
        if len(pert):
            distance, _ = cKDTree(data.coordinates[pert]).query(data.coordinates[ids])
            q[ids, -3] = np.log1p(distance)
            q[ids, -2] = 1
    return q, names


def build_modules(x, types, train, n_modules):
    """Training control receivers only; remove cell-type means before correlations."""
    residual = x[train].astype(np.float64).copy()
    for t in np.unique(types[train]):
        selected = types[train] == t
        residual[selected] -= residual[selected].mean(0)
    norm = np.sqrt((residual ** 2).sum(0))
    standardized = residual / np.maximum(norm, 1e-12)
    corr = np.clip(standardized.T @ standardized, -1, 1)
    distance = np.clip(1 - corr, 0, 2)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    if x.shape[1] < 2:
        return np.zeros(x.shape[1], dtype=np.int64)
    tree = linkage(squareform(distance, checks=False), method="average")
    return fcluster(tree, t=min(n_modules, x.shape[1]), criterion="maxclust").astype(np.int64) - 1


def mask_rows(n_rows, n_genes, rng, ratio, modules=None):
    result = np.zeros((n_rows, n_genes), dtype=bool)
    count = max(1, min(n_genes, int(round(ratio * n_genes))))
    for row in result:
        if modules is None:
            row[rng.choice(n_genes, count, replace=False)] = True
        else:
            # Mask whole modules until reaching the requested minimum gene count.
            for m in rng.permutation(np.unique(modules)):
                row[modules == m] = True
                if row.sum() >= count:
                    break
    return result


def altered_graph(data, graph, seed, conditional=False, rewire=False, report_receivers=None):
    """Fixed validation null; preserve edge count/weights, tissue and no-self constraint."""
    rng = np.random.default_rng(seed)
    strata = {}
    for i in range(len(data.x)):
        key = (data.tissue[i], int(data.types[i]), data.perturbation[i]) if conditional else (data.tissue[i],)
        strata.setdefault(key, []).append(i)
    pools = {k: np.asarray(v, dtype=np.int64) for k, v in strata.items()}
    mapping = np.arange(len(data.x))
    for ids in pools.values():
        mapping[ids] = rng.permutation(ids)
    output = NullGraph(data, graph, seed, mapping, pools, rewire)
    changed, total = 0, 0
    for receiver in (range(len(graph)) if report_receivers is None else report_receivers):
        src, weight = graph[receiver]
        new_src, _ = output[receiver]
        changed += int((new_src != src).sum())
        total += len(src)
    return output, changed / max(total, 1)
