"""
Fraud Ring Detector — Community detection on transaction graph.

Algorithm:
  1. Convert PyG graph → NetworkX undirected graph
  2. Run Louvain community detection  (BFS-based modularity maximisation, O(n log n))
  3. Filter communities by density threshold  → fraud rings
  4. Inside each ring, find money mule via betweenness centrality  (Dijkstra, O(VE))

DSA connection:
  - Louvain internally runs BFS to expand communities
  - Betweenness centrality = fraction of all-pairs shortest paths through a node
  - Both are real graph algorithm applications, not just ML
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import torch
from community import best_partition          # python-louvain
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx


# ── data class ───────────────────────────────────────────────────────────────

@dataclass
class FraudRing:
    ring_id      : int
    nodes        : list[int]
    density      : float
    money_mule   : Optional[int]
    betweenness  : dict[int, float] = field(default_factory=dict)
    total_amount : float = 0.0
    status       : str = "active"   # "active" | "resolved"

    def to_dict(self) -> dict:
        return {
            "ring_id"     : self.ring_id,
            "nodes"       : self.nodes,
            "size"        : len(self.nodes),
            "density"     : round(self.density, 4),
            "money_mule"  : self.money_mule,
            "total_amount": round(self.total_amount, 2),
            "status"      : self.status,
        }


# ── core detector ─────────────────────────────────────────────────────────────

class RingDetector:
    def __init__(
        self,
        min_ring_size      : int   = 3,
        density_threshold  : float = 0.60,
        verbose            : bool  = True,
    ):
        self.min_ring_size     = min_ring_size
        self.density_threshold = density_threshold
        self.verbose           = verbose

    # ── graph conversion ──────────────────────────────────────────────────────

    def _pyg_to_nx(self, data: HeteroData) -> nx.Graph:
        """
        Convert HeteroData → simple undirected NetworkX graph.
        We only use the (user, sends, txn) edge type.
        """
        G = nx.Graph()
        edge_index = data["user", "sends", "txn"].edge_index
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()

        # Offset txn node IDs to avoid clash with user node IDs
        n_users = data["user"].num_nodes
        for u, t in zip(src, dst):
            G.add_edge(u, t + n_users)

        return G

    def _nx_from_edges(self, edges: list[tuple]) -> nx.Graph:
        """Build nx.Graph directly from (src, dst) edge list."""
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    # ── community detection ───────────────────────────────────────────────────

    def _detect_communities(self, G: nx.Graph) -> list[set]:
        """
        Louvain community detection.
        Returns list of node-sets (communities).
        """
        partition = best_partition(G, random_state=42)
        communities: dict[int, set] = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(node)
        return list(communities.values())

    # ── ring analysis ─────────────────────────────────────────────────────────

    def _analyse_ring(
        self,
        ring_id  : int,
        community: set,
        G        : nx.Graph,
        amounts  : Optional[dict] = None,
    ) -> FraudRing:
        sub = G.subgraph(community)
        density = nx.density(sub)

        # Money mule = node with highest betweenness centrality
        betweenness = nx.betweenness_centrality(sub, normalized=True)
        mule = max(betweenness, key=betweenness.get)

        total_amount = sum(amounts.get(n, 0) for n in community) if amounts else 0.0

        return FraudRing(
            ring_id      = ring_id,
            nodes        = sorted(community),
            density      = density,
            money_mule   = mule,
            betweenness  = {k: round(v, 4) for k, v in betweenness.items()},
            total_amount = total_amount,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        data   : HeteroData,
        amounts: Optional[dict] = None,
    ) -> list[FraudRing]:
        """
        Full detection pipeline on a PyG HeteroData graph.
        Returns list of FraudRing objects.
        """
        t0 = time.time()
        G  = self._pyg_to_nx(data)

        if self.verbose:
            print(f"NetworkX graph: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges")

        communities = self._detect_communities(G)
        rings       = []
        ring_id     = 1

        for community in communities:
            if len(community) < self.min_ring_size:
                continue
            sub     = G.subgraph(community)
            density = nx.density(sub)
            if density < self.density_threshold:
                continue

            ring = self._analyse_ring(ring_id, community, G, amounts)
            rings.append(ring)
            ring_id += 1

        elapsed = time.time() - t0
        if self.verbose:
            print(f"Detected {len(rings)} fraud rings in {elapsed:.2f}s")
            for r in rings:
                print(f"  Ring #{r.ring_id}: {len(r.nodes)} nodes | "
                      f"density={r.density:.3f} | mule={r.money_mule} | "
                      f"₹{r.total_amount:,.0f}")

        return rings

    def detect_from_edges(
        self,
        edges  : list[tuple],
        amounts: Optional[dict] = None,
    ) -> list[FraudRing]:
        """
        Detect rings directly from an edge list.
        Useful for streaming (incremental updates).

        edges = [(src_node_id, dst_node_id), ...]
        """
        G  = self._nx_from_edges(edges)
        cs = self._detect_communities(G)

        rings    = []
        ring_id  = 1
        for c in cs:
            if len(c) < self.min_ring_size:
                continue
            sub = G.subgraph(c)
            if nx.density(sub) < self.density_threshold:
                continue
            rings.append(self._analyse_ring(ring_id, c, G, amounts))
            ring_id += 1

        return rings


# ── streaming incremental graph ───────────────────────────────────────────────

class IncrementalGraph:
    """
    Maintains a live edge list as Kafka messages arrive.
    Periodically re-runs RingDetector.
    """

    def __init__(self, detect_interval: int = 100):
        self.edges            : list[tuple]  = []
        self.amounts          : dict[int, float] = {}
        self.detect_interval  = detect_interval
        self.txn_count        = 0
        self.detector         = RingDetector(verbose=False)
        self.latest_rings     : list[FraudRing] = []

    def add_transaction(self, src: int, dst: int, amount: float):
        self.edges.append((src, dst))
        self.amounts[dst] = self.amounts.get(dst, 0) + amount
        self.txn_count += 1

        if self.txn_count % self.detect_interval == 0:
            self.latest_rings = self.detector.detect_from_edges(
                self.edges[-5000:],   # sliding window of last 5000 edges
                self.amounts,
            )

    def get_rings(self) -> list[dict]:
        return [r.to_dict() for r in self.latest_rings]


# ── cli ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph_path = Path("data/processed/graph.pt")
    if not graph_path.exists():
        print("Graph not found. Run: python data/graph_builder.py")
        exit(1)

    data     = torch.load(graph_path)
    detector = RingDetector(min_ring_size=3, density_threshold=0.60)
    rings    = detector.detect(data)

    if rings:
        print("\n── Ring summary ──")
        for r in rings:
            print(r.to_dict())
    else:
        print("No fraud rings detected with current thresholds.")
