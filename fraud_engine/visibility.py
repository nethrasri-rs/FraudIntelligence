import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple
from .config import config

class VisibilityEstimator:
    """
    Computes per-node Missingness / Visibility score M.
    Core concept:
    A bank only sees edges inside its own walls.
    When a customer interacts with external non-observable counterparties,
    the graph model operates with incomplete topology.

    M is bounded in [0, 1]:
      0.0 -> Perfect visibility (all counterparties are internal to PayPal / home bank).
      1.0 -> Zero visibility (all counterparties are external / invisible).
    """

    def __init__(self, home_institution: str = None):
        self.home_institution = home_institution or config.HOME_INSTITUTION_ID
        # Baseline expected degrees per account tier
        self.expected_degrees = {
            "personal": 4.5,
            "merchant": 25.0,
            "high_volume": 80.0
        }

    def compute_counterparty_coverage(self, G: nx.Graph, node_id: str) -> float:
        """
        Counterparty-coverage ratio based Missingness M.
        M = 1 - (|internal counterparties| / |total counterparties|)
        Applies Laplace smoothing for low degree nodes.
        """
        if node_id not in G:
            return 0.5  # Unknown node default missingness

        neighbors = list(G.neighbors(node_id))
        if not neighbors:
            return 0.5

        total_counterparties = 0
        internal_counterparties = 0

        for nbr in neighbors:
            # Check edge or node institution
            nbr_inst = G.nodes[nbr].get("institution", self.home_institution)
            total_counterparties += 1
            if nbr_inst == self.home_institution:
                internal_counterparties += 1

        # Laplace smoothing (pseudo-count = 1)
        # coverage in [0, 1], M = 1 - coverage
        coverage = (internal_counterparties + 1.0) / (total_counterparties + 2.0)
        M = 1.0 - coverage
        return float(np.clip(M, 0.0, 1.0))

    def compute_degree_gap(self, G: nx.Graph, node_id: str, account_tier: str = "personal") -> float:
        """
        Degree-gap estimate:
        Observed degree vs expected historical degree.
        If observed degree is much lower than expected activity indicates,
        external activity is likely hidden.
        """
        if node_id not in G:
            return 0.5

        observed_degree = G.degree(node_id)
        expected = self.expected_degrees.get(account_tier, 5.0)

        gap = max(0.0, (expected - observed_degree) / (expected + 1.0))
        return float(np.clip(gap, 0.0, 1.0))

    def compute_node_missingness(self, G: nx.Graph, node_id: str, account_tier: str = "personal") -> Dict[str, float]:
        """
        Combines Counterparty Coverage (primary) and Degree Gap (secondary)
        into the unified visibility/missingness signal M.
        """
        m_coverage = self.compute_counterparty_coverage(G, node_id)
        m_gap = self.compute_degree_gap(G, node_id, account_tier)

        # Weighted combination: 80% coverage ratio, 20% degree gap
        m_combined = 0.8 * m_coverage + 0.2 * m_gap

        return {
            "m_score": float(np.round(m_combined, 4)),
            "m_coverage": float(np.round(m_coverage, 4)),
            "m_gap": float(np.round(m_gap, 4)),
            "observable_ratio": float(np.round(1.0 - m_combined, 4))
        }

visibility_estimator = VisibilityEstimator()
