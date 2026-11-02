import re
import requests
import json
from typing import Dict, Any, List, Tuple
from .config import config
from .types import SubgraphData, GroundedSummary

class CitationChecker:
    """
    Verifies that every entity or transaction ID cited in the LLM explanation
    actually exists in the verified Subgraph evidence pack.
    Rejects sentences containing hallucinations.
    """

    def __init__(self):
        # Regex to capture cited entities like [evt_123], [acc_xyz], [dev_abc], etc.
        self.entity_pattern = re.compile(r'\[([a-zA-Z0-9_\-]+)\]')

    def verify_explanation(self, summary_text: str, subgraph: Dict[str, Any]) -> GroundedSummary:
        valid_nodes = set()
        valid_edges = set()

        if subgraph:
            for node in subgraph.get("nodes", []):
                valid_nodes.add(str(node.get("id")))
                if "device_id" in node:
                    valid_nodes.add(f"dev_{node['device_id']}")
            for edge in subgraph.get("edges", []):
                if "event_id" in edge:
                    valid_edges.add(str(edge["event_id"]))
                if "key" in edge:
                    valid_edges.add(str(edge["key"]))

        sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
        verified_sentences = []
        cited_nodes = set()
        cited_edges = set()
        rejected_sentences = []

        for sent in sentences:
            if not sent:
                continue
            matches = self.entity_pattern.findall(sent)
            sent_valid = True
            for entity in matches:
                if entity in valid_nodes:
                    cited_nodes.add(entity)
                elif entity in valid_edges:
                    cited_edges.add(entity)
                else:
                    # Hallucinated ID not found in evidence
                    sent_valid = False
                    rejected_sentences.append(f"Rejected sentence citing unknown entity '{entity}': \"{sent}\"")
                    break

            if sent_valid:
                verified_sentences.append(sent)

        cleaned_summary = " ".join(verified_sentences)
        is_verified = (len(rejected_sentences) == 0) and len(cleaned_summary) > 0

        return GroundedSummary(
            summary_text=cleaned_summary if cleaned_summary else summary_text,
            cited_nodes=list(cited_nodes),
            cited_edges=list(cited_edges),
            is_verified=is_verified,
            rejection_reason="; ".join(rejected_sentences) if rejected_sentences else None
        )

class GroundedExplainer:
    """
    Produces evidence-grounded analyst explanations citing explicit graph tokens.
    Never invents scores; only synthesizes structured evidence.
    """

    def __init__(self):
        self.checker = CitationChecker()
        self.api_key = config.GEMINI_API_KEY

    def _build_deterministic_grounded_summary(
        self,
        account_id: str,
        reason_codes: List[str],
        p_hybrid: float,
        visibility_m: float,
        subgraph: Dict[str, Any]
    ) -> str:
        """
        Deterministic, hallucination-free grounded generator.
        Ensures 100% citation compliance even when offline.
        """
        lines = []
        lines.append(f"Case Analysis for account [{account_id}] (Risk: {p_hybrid:.2f}, Missingness M: {visibility_m:.2f}).")

        # 1. Reason code summary
        if reason_codes:
            lines.append(f"Triggered detection signals: {', '.join(reason_codes)}.")

        # 2. Graph topology details citing exact nodes
        nodes = subgraph.get("nodes", []) if subgraph else []
        edges = subgraph.get("edges", []) if subgraph else []

        devices = [n["id"] for n in nodes if n.get("type") == "device"]
        accounts = [n["id"] for n in nodes if n.get("type") == "account" and n["id"] != account_id]

        if devices:
            lines.append(f"Account [{account_id}] shares device token [{devices[0]}] across multiple identity entities.")

        if accounts:
            cited_accs = ", ".join(f"[{a}]" for a in accounts[:3])
            lines.append(f"Immediate counterparty interactions include {cited_accs}.")

        if edges:
            sample_edge = edges[0]
            evt_id = sample_edge.get("event_id", sample_edge.get("key"))
            amt = sample_edge.get("amount", 0)
            lines.append(f"Primary transaction [{evt_id}] transferred ${amt:.2f} within the active window.")

        # 3. Missingness notice
        if visibility_m >= 0.45:
            lines.append(
                f"Notice: Visibility score M={visibility_m:.2f} indicates external banking blind-spots. "
                f"Abstain decision recommended pending cross-institution telemetry."
            )

        return " ".join(lines)

    def generate_explanation(
        self,
        account_id: str,
        reason_codes: List[str],
        p_hybrid: float,
        visibility_m: float,
        subgraph: Dict[str, Any]
    ) -> GroundedSummary:
        # Generate base text
        raw_text = self._build_deterministic_grounded_summary(
            account_id, reason_codes, p_hybrid, visibility_m, subgraph
        )

        # Run through verification checker
        verified = self.checker.verify_explanation(raw_text, subgraph)
        return verified

grounded_explainer = GroundedExplainer()
