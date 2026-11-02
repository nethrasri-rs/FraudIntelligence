import numpy as np
from typing import Dict, Any, List, Tuple
from ..types import TransactionEvent, DecisionType
from ..config import config

class MissingnessAwareCalibrator:
    """
    Multi-Branch Fusion & Missingness Calibrator
    
    Computes P_hybrid by dynamically weighting:
      - Tabular score (P_tab)
      - Causal 1D CNN score (P_cnn)
      - GraphSAGE score (P_gnn)
      - Missingness score (M)
      
    Dynamic weighting W(M):
      When M is LOW (< 0.2, high visibility inside PayPal/bank):
        - Model heavily trusts the Graph topology (P_gnn weight: 0.50).
      When M is HIGH (> 0.5, significant counterparty blindness):
        - Model down-weights the Graph topology (P_gnn weight decays to 0.15).
        - Model relies more on behavioral velocity (P_cnn, P_tab).
        
    Three-way Decision:
      - BLOCK: High risk P_hybrid >= THRESHOLD_BLOCK (0.75).
      - ALLOW: Low risk P_hybrid < THRESHOLD_ALLOW (0.30) AND sufficient visibility M < THRESHOLD_INSUFFICIENT_VISIBILITY (0.45).
      - INSUFFICIENT VISIBILITY: Low/moderate apparent graph risk BUT high missingness (M >= 0.45),
        or ambiguous border region. The engine abstains rather than making an ungrounded false-allow!
    """

    def __init__(self):
        self.thresh_block = config.THRESHOLD_BLOCK
        self.thresh_allow = config.THRESHOLD_ALLOW
        self.thresh_visibility = config.THRESHOLD_INSUFFICIENT_VISIBILITY

    def compute_dynamic_weights(self, m: float) -> np.ndarray:
        """
        Dynamically adjusts branch weights based on M:
        [w_tab, w_cnn, w_gnn, w_m]
        """
        # GNN weight decays smoothly as M increases: w_gnn = 0.50 * (1 - 0.7 * M)
        w_gnn = max(0.12, 0.50 * (1.0 - 0.75 * m))
        # Tabular and CNN weights compensate when graph is unobservable
        w_tab = 0.25 + (0.20 * m)
        w_cnn = 0.25 + (0.20 * m)
        w_m = 0.10

        # Normalize to sum to 1.0 (excluding M interaction term)
        scale = (w_tab + w_cnn + w_gnn)
        return np.array([w_tab / scale, w_cnn / scale, w_gnn / scale, w_m], dtype=np.float32)

    def calibrate(
        self,
        p_tab: float,
        p_cnn: float,
        p_gnn: float,
        m: float,
        reason_codes: List[str]
    ) -> Dict[str, Any]:
        weights = self.compute_dynamic_weights(m)
        w_tab, w_cnn, w_gnn, w_m = weights

        # Logistic logit fusion
        raw_weighted_prob = (w_tab * p_tab) + (w_cnn * p_cnn) + (w_gnn * p_gnn)
        
        # Calibration shift based on M-bucket:
        # In high-missingness buckets, standard models underpredict risk due to absent edges.
        # Calibrator adjusts the logit:
        logit = np.log(raw_weighted_prob / (1.0 - raw_weighted_prob + 1e-7))
        # Upward risk adjustment for missingness uncertainty
        m_adjustment = w_m * (m - 0.25)
        p_hybrid = 1.0 / (1.0 + np.exp(-(logit + m_adjustment)))
        p_hybrid = float(np.clip(p_hybrid, 0.01, 0.99))

        # Decision Logic (Three-Way Decision)
        if p_hybrid >= self.thresh_block:
            decision = DecisionType.BLOCK
            if "HIGH_HYBRID_RISK" not in reason_codes:
                reason_codes.append("HIGH_HYBRID_RISK")
        elif m >= self.thresh_visibility and p_hybrid < self.thresh_block:
            # SIGNATURE FEATURE: Abstain due to insufficient visibility
            # The network looks clean or moderate only because external edges cannot be seen!
            decision = DecisionType.INSUFFICIENT_VISIBILITY
            reason_codes.append("INSUFFICIENT_NETWORK_VISIBILITY")
            reason_codes.append(f"MISSINGNESS_SCORE_{round(m, 2)}")
        elif p_hybrid < self.thresh_allow:
            decision = DecisionType.ALLOW
        else:
            decision = DecisionType.REVIEW
            reason_codes.append("BORDERLINE_SUSPICIOUS_VELOCITY")

        return {
            "p_hybrid": float(np.round(p_hybrid, 4)),
            "decision": decision,
            "weights": {
                "w_tab": float(np.round(w_tab, 3)),
                "w_cnn": float(np.round(w_cnn, 3)),
                "w_gnn": float(np.round(w_gnn, 3)),
                "w_m": float(np.round(w_m, 3))
            },
            "reason_codes": reason_codes
        }

calibrator = MissingnessAwareCalibrator()
