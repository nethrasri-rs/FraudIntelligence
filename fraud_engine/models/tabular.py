import numpy as np
from typing import Dict, Any, List
from ..types import TransactionEvent
from ..graph_builder import global_graph

class TabularFraudModel:
    """
    Branch 1: Tabular Feature Extraction & Scoring (P_tabular)
    Features extracted:
    - Transaction amount & log-amount
    - Rolling 1h / 24h count & sum
    - Outflow to inflow ratio (fund dissipation)
    - Device sharing count
    - Cross-border / external institution flag
    """

    def __init__(self):
        # Pre-calibrated weights for tabular features
        self.feature_weights = np.array([
            0.45,   # log_amount
            0.60,   # velocity_1h
            0.75,   # dissipation_ratio (outflow / inflow)
            0.85,   # shared_device_count
            0.50    # external_institution_flag
        ], dtype=np.float32)
        self.bias = -2.8

    def extract_features(self, event: TransactionEvent) -> np.ndarray:
        # 1. log amount
        log_amt = np.log1p(max(0.0, event.amount)) / 10.0

        # 2. velocity (number of transactions in last 5 minutes)
        recent_txs = global_graph.get_recent_transactions(event.source_account, window_seconds=300.0)
        velocity_count = min(len(recent_txs), 10) / 10.0

        # 3. fund dissipation ratio
        inflow = sum(tx.get("amount", 0) for tx in recent_txs if tx.get("direction") == "IN")
        outflow = sum(tx.get("amount", 0) for tx in recent_txs if tx.get("direction") == "OUT") + event.amount
        dissipation = (outflow / (inflow + 1.0)) if inflow > 0 else (1.0 if outflow > 1000 else 0.2)
        dissipation = min(dissipation / 5.0, 1.0)

        # 4. device sharing
        shared_devices = 0
        if event.device_id:
            shared_accounts = global_graph.get_accounts_sharing_device(event.device_id)
            shared_devices = min(len(shared_accounts), 5) / 5.0

        # 5. external counterparty
        is_external = 1.0 if event.target_institution != event.source_institution else 0.0

        return np.array([log_amt, velocity_count, dissipation, shared_devices, is_external], dtype=np.float32)

    def predict_proba(self, event: TransactionEvent) -> float:
        feats = self.extract_features(event)
        logit = float(np.dot(self.feature_weights, feats) + self.bias)
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(p, 0.01, 0.99))

tabular_model = TabularFraudModel()
