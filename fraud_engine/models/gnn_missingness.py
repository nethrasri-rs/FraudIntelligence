import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from ..types import TransactionEvent, SubgraphData
from ..graph_builder import global_graph
from ..visibility import visibility_estimator

class MissingnessAwareGraphSAGE:
    """
    Branch 3: Missingness-Aware Graph Neural Network (GraphSAGE with Visibility Attention)
    
    Key innovations:
    1. Edge-Dropout Robustness: Simulates partial cross-bank observation during training/inference.
    2. Visibility-Scaled Message Passing: Message from neighbor v to u is scaled by
       v's visibility (1 - M_v). If a neighbor is external or partially observable,
       its unverified message weight is down-scaled.
    3. Structural Mule/Cycle Detection:
       - Fan-in: Multiple accounts concentrating funds into one node.
       - Fan-out: One node dispersing funds rapidly to multiple intermediaries.
       - Cycles: Directed loops (A -> B -> C -> A) characteristic of smurfing & layering.
    """

    def __init__(self, edge_dropout_rate: float = 0.0, use_visibility_attention: bool = True):
        self.edge_dropout_rate = edge_dropout_rate
        self.use_visibility_attention = use_visibility_attention
        # Dimension of node embeddings
        self.dim = 16
        # Initial projection weights
        np.random.seed(42)
        self.W_self = np.random.randn(self.dim, self.dim).astype(np.float32) * 0.1
        self.W_neigh = np.random.randn(self.dim, self.dim).astype(np.float32) * 0.1

    def _get_node_features(self, G: nx.MultiDiGraph, node: str) -> np.ndarray:
        in_deg = G.in_degree(node) if G.has_node(node) else 0
        out_deg = G.out_degree(node) if G.has_node(node) else 0
        inst = G.nodes[node].get("institution", "PAYPAL_HOME_BANK") if G.has_node(node) else "OTHER"
        is_internal = 1.0 if inst == "PAYPAL_HOME_BANK" else 0.0

        feats = np.zeros(self.dim, dtype=np.float32)
        feats[0] = min(in_deg, 20) / 20.0
        feats[1] = min(out_deg, 20) / 20.0
        feats[2] = is_internal
        # Fan ratio: (out - in) / (out + in + 1)
        feats[3] = (out_deg - in_deg) / (out_deg + in_deg + 1e-4)
        return feats

    def detect_graph_patterns(self, G: nx.MultiDiGraph, center_node: str) -> Dict[str, float]:
        """
        Analyzes topological laundering motifs in center_node's ego-graph:
        - fan_in_intensity: high in-degree, low out-degree
        - fan_out_intensity: high out-degree, low in-degree
        - cycle_intensity: participation in directed cycles
        """
        if not G.has_node(center_node):
            return {"fan_in": 0.0, "fan_out": 0.0, "cycle": 0.0, "ring_density": 0.0}

        in_deg = G.in_degree(center_node)
        out_deg = G.out_degree(center_node)

        # Fan-in pattern (layering / mule aggregator)
        fan_in = 1.0 if (in_deg >= 4 and out_deg <= 2) else min(in_deg / 6.0, 1.0) * 0.6

        # Fan-out pattern (originator / smurfing)
        fan_out = 1.0 if (out_deg >= 4 and in_deg <= 2) else min(out_deg / 6.0, 1.0) * 0.6

        # Cycle / Ring detection (simple paths of length 2 or 3 back to center_node)
        cycle_found = 0.0
        try:
            # Check if any successor leads back to center_node
            successors = list(G.successors(center_node))
            for s in successors:
                if s == center_node:
                    continue
                if G.has_edge(s, center_node):
                    cycle_found = 1.0
                    break
                # 3-hop cycle
                for s2 in G.successors(s):
                    if s2 != s and s2 != center_node and G.has_edge(s2, center_node):
                        cycle_found = 1.0
                        break
                if cycle_found > 0:
                    break
        except Exception:
            cycle_found = 0.0

        # Ring density
        neighbors = set(G.predecessors(center_node)).union(set(G.successors(center_node)))
        density = 0.0
        if len(neighbors) > 1:
            sub = G.subgraph(neighbors.union({center_node}))
            density = min(nx.density(sub) * 2.0, 1.0)

        return {
            "fan_in": float(fan_in),
            "fan_out": float(fan_out),
            "cycle": float(cycle_found),
            "ring_density": float(density)
        }

    def predict_proba(
        self,
        event: TransactionEvent,
        simulated_drop_rate: Optional[float] = None,
        is_baseline_standard: bool = False,
        graph_override: Optional[nx.MultiDiGraph] = None
    ) -> Dict[str, Any]:
        """
        Calculates GraphSAGE risk score P_GNN.
        If is_baseline_standard=True, acts as the standard naive GraphSAGE:
          - Does not scale by visibility (treats missing edges as truly non-existent).
          - Prone to false negatives when edges to external banks are dropped.
        """
        center_node = event.source_account
        drop_rate = simulated_drop_rate if simulated_drop_rate is not None else self.edge_dropout_rate

        # Get working view of graph, applying edge-drop if simulated
        G_active = graph_override if graph_override is not None else global_graph.G
        if drop_rate > 0.0:
            # Apply edge dropout (drop a fraction of external or random edges)
            sampled_G = nx.MultiDiGraph()
            sampled_G.add_nodes_from(G_active.nodes(data=True))
            for u, v, k, d in G_active.edges(keys=True, data=True):
                # External edges are dropped with higher probability to simulate banking walls
                is_ext = (G_active.nodes[u].get("institution") != G_active.nodes[v].get("institution"))
                effective_drop = min(drop_rate * 1.5, 0.95) if is_ext else drop_rate
                if np.random.rand() > effective_drop:
                    sampled_G.add_edge(u, v, key=k, **d)
            G_eval = sampled_G
        else:
            G_eval = G_active

        # Extract structural patterns
        patterns = self.detect_graph_patterns(G_eval, center_node)

        # Message passing aggregation
        h_self = self._get_node_features(G_eval, center_node)
        
        # Aggregate neighbor features with visibility attention
        if G_eval.has_node(center_node):
            neighbors = list(G_eval.successors(center_node)) + list(G_eval.predecessors(center_node))
        else:
            neighbors = []
        if neighbors:
            nbr_feats = []
            for nbr in neighbors:
                h_nbr = self._get_node_features(G_eval, nbr)
                if not is_baseline_standard and self.use_visibility_attention:
                    # Scale neighbor message by that neighbor's visibility
                    nbr_m = visibility_estimator.compute_counterparty_coverage(G_eval, nbr)
                    visibility_weight = 1.0 - (0.7 * nbr_m)
                    h_nbr = h_nbr * visibility_weight
                nbr_feats.append(h_nbr)
            h_neigh_agg = np.mean(nbr_feats, axis=0)
        else:
            h_neigh_agg = np.zeros(self.dim, dtype=np.float32)

        # Combine
        embedding = np.tanh(np.dot(h_self, self.W_self) + np.dot(h_neigh_agg, self.W_neigh))

        # Graph risk logit
        motif_risk = (
            patterns["cycle"] * 1.8 +
            patterns["fan_in"] * 1.2 +
            patterns["fan_out"] * 1.2 +
            patterns["ring_density"] * 0.8
        )
        embed_risk = float(np.mean(embedding[:4]))

        logit = motif_risk + embed_risk - 1.2
        p_gnn = float(1.0 / (1.0 + np.exp(-logit)))
        p_gnn = float(np.clip(p_gnn, 0.01, 0.99))

        return {
            "p_gnn": p_gnn,
            "patterns": patterns,
            "node_degree": G_eval.degree(center_node) if G_eval.has_node(center_node) else 0
        }

gnn_model = MissingnessAwareGraphSAGE()
