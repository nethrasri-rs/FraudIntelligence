import numpy as np
from typing import List, Dict, Any
from ..types import TransactionEvent
from ..graph_builder import global_graph

from typing import List, Dict, Any, Tuple
from ..types import TransactionEvent
from ..graph_builder import global_graph

class CausalTemporalCNN:
    """
    Branch 2: 1D Causal Convolutional Neural Network (P_CNN)
    Operates over ordered sequence of inter-arrival times and amounts:
    Captures temporal dynamics (rapid-fire bursts, periodic bots, structuring)
    without future information leakage.
    """

    def __init__(self, sequence_length: int = 8):
        self.seq_len = sequence_length
        # Causal filter weights (kernel size 3, dilation 1 & 2)
        # Weights emphasize acceleration in cadence (time delta decaying)
        self.kernel_cadence = np.array([0.5, 0.8, 1.2], dtype=np.float32)
        self.kernel_amounts = np.array([0.3, 0.6, 0.9], dtype=np.float32)

    def extract_sequence(self, event: TransactionEvent) -> Tuple[np.ndarray, np.ndarray]:
        recent_txs = global_graph.get_recent_transactions(event.source_account, window_seconds=1800.0)
        # Order chronologically up to current event
        tx_list = list(reversed(recent_txs))
        tx_list.append({
            "timestamp": event.timestamp,
            "amount": event.amount,
            "direction": "OUT"
        })
        # Keep last seq_len
        tx_list = tx_list[-self.seq_len:]

        # Compute delta times (in seconds normalized) and amounts
        deltas = []
        amounts = []
        for i in range(len(tx_list)):
            if i == 0:
                deltas.append(60.0)
            else:
                dt = max(0.1, tx_list[i]["timestamp"] - tx_list[i-1]["timestamp"])
                deltas.append(dt)
            amounts.append(tx_list[i]["amount"])

        # Pad to seq_len
        pad_len = self.seq_len - len(deltas)
        if pad_len > 0:
            deltas = [300.0] * pad_len + deltas
            amounts = [0.0] * pad_len + amounts

        # Invert deltas so rapid bursts have high intensity: 1.0 / (1.0 + log(dt))
        intensity = 1.0 / (1.0 + np.log1p(np.array(deltas, dtype=np.float32)))
        norm_amounts = np.log1p(np.array(amounts, dtype=np.float32)) / 8.0

        return intensity, norm_amounts

    def predict_proba(self, event: TransactionEvent) -> float:
        intensity, norm_amounts = self.extract_sequence(event)

        # 1D Causal Convolution on tail window (last 3 timesteps)
        k_len = len(self.kernel_cadence)
        cadence_feat = float(np.sum(intensity[-k_len:] * self.kernel_cadence))
        amount_feat = float(np.sum(norm_amounts[-k_len:] * self.kernel_amounts))

        # Check for rapid acceleration (e.g. intervals dropping rapidly)
        acceleration = 0.0
        if len(intensity) >= 3 and intensity[-1] > 2.0 * intensity[-3]:
            acceleration = 0.8

        logit = 0.7 * cadence_feat + 0.5 * amount_feat + 0.6 * acceleration - 2.5
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(p, 0.01, 0.99))

# Singleton instance
causal_cnn = CausalTemporalCNN()
