"""
data_utils.py
Materials Project fetching + crystal-to-graph conversion, shared by both
pretraining scripts so dataset construction (cutoff radius, fields,
sample size) can't silently drift between experiments.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from mp_api.client import MPRester
from sklearn.neighbors import NearestNeighbors

CUTOFF_RADIUS = 4.0
MP_FIELDS = ["material_id", "structure", "formula_pretty", "band_gap"]


def fetch_materials(api_key, target_n=3600, max_n_sites=80, chunk_size=100, num_chunks=40):
    """
    Pull crystal structures from Materials Project, capped at max_n_sites
    atoms per structure. Only requests the fields actually used downstream
    (requesting fields=None pulls every available field and is much slower).
    """
    entries = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            num_elements=(1, 8),
            fields=MP_FIELDS,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
        for doc in docs:
            if doc.structure and len(doc.structure) <= max_n_sites:
                entries.append(doc)
                if len(entries) >= target_n:
                    break
    return entries


def structure_to_pyg_data(structure, cutoff: float = CUTOFF_RADIUS) -> Data:
    """
    Convert a pymatgen Structure into a PyG graph: atoms as nodes, bonds
    within `cutoff` Angstroms as edges (correctly handling periodic images
    via get_all_neighbors). Falls back to k-NN for any isolated atoms with
    no neighbor inside the cutoff.
    """
    atomic_nums = [site.specie.number for site in structure.sites]
    x = torch.tensor(atomic_nums, dtype=torch.float).view(-1, 1)

    edge_list, edge_attr = [], []
    neighbors = structure.get_all_neighbors(cutoff)
    for i, site_neighbors in enumerate(neighbors):
        for nb in site_neighbors:
            edge_list.append([i, nb.index])
            edge_attr.append([nb.nn_distance])

    if len(edge_list) == 0:
        coords = structure.cart_coords
        nbrs = NearestNeighbors(n_neighbors=min(6, len(structure))).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        for i in range(len(structure)):
            for k_idx, j in enumerate(indices[i]):
                if i == j:
                    continue
                edge_list.append([i, int(j)])
                edge_attr.append([float(distances[i, k_idx])])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def augment_graph(graph: Data, drop_node_prob: float = 0.15) -> Data:
    """
    Node-dropping augmentation for contrastive learning.

    Fixes:
    1. Prevents tiny graphs (<2 nodes) after augmentation.
    2. Preserves edge_attr alignment.
    3. Preserves band_gap and any future graph attributes.
    """

    num_nodes = graph.num_nodes

    # Tiny graphs can't safely be augmented
    if num_nodes <= 2:
        return graph.clone()

    mask = torch.rand(num_nodes) > drop_node_prob

    # Ensure at least 2 nodes survive
    if mask.sum() < 2:
        keep = torch.randperm(num_nodes)[:2]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[keep] = True

    subset = torch.where(mask)[0]

    edge_index, edge_attr = subgraph(
        subset,
        graph.edge_index,
        edge_attr=graph.edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    x = graph.x[subset]

    new_graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Preserve all extra attributes
    for key in graph.keys():
        if key not in ["x", "edge_index", "edge_attr"]:
            setattr(new_graph, key, getattr(graph, key))

    return new_graph