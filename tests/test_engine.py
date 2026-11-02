import pytest
import time
import networkx as nx
from fraud_engine.types import TransactionEvent, EventType, DecisionType
from fraud_engine.visibility import visibility_estimator
from fraud_engine.fast_filter import fast_filter
from fraud_engine.graph_builder import global_graph, TransactionGraph
from fraud_engine.models.tabular import tabular_model
from fraud_engine.models.causal_cnn import causal_cnn
from fraud_engine.models.gnn_missingness import gnn_model
from fraud_engine.models.fusion import calibrator
from fraud_engine.explainer import CitationChecker
from fraud_engine.pipeline import engine_pipeline
from storage.db import db

def test_visibility_estimator():
    G = nx.MultiDiGraph()
    # Node with 1 internal and 3 external counterparties
    G.add_node("acc_test", institution="PAYPAL_HOME_BANK")
    G.add_node("acc_int", institution="PAYPAL_HOME_BANK")
    G.add_node("acc_ext_1", institution="OTHER_BANK")
    G.add_node("acc_ext_2", institution="OTHER_BANK")
    G.add_node("acc_ext_3", institution="OTHER_BANK")

    G.add_edge("acc_test", "acc_int")
    G.add_edge("acc_test", "acc_ext_1")
    G.add_edge("acc_test", "acc_ext_2")
    G.add_edge("acc_test", "acc_ext_3")

    m_info = visibility_estimator.compute_node_missingness(G, "acc_test")
    # Coverage should be low because 3 out of 4 counterparties are external -> M is high (>0.50)
    assert m_info["m_score"] > 0.50
    assert m_info["observable_ratio"] < 0.50

def test_fast_filter_shared_device():
    global_graph.clear()
    # 3 accounts on same device
    for i in range(3):
        ev = TransactionEvent(
            source_account=f"acc_dev_{i}",
            target_account="acc_target",
            amount=100.0,
            device_id="dev_shared_123"
        )
        global_graph.add_transaction(ev)

    test_ev = TransactionEvent(
        source_account="acc_dev_4",
        target_account="acc_target",
        amount=200.0,
        device_id="dev_shared_123"
    )
    res = fast_filter.evaluate(test_ev)
    assert res.latency_ms < 25.0
    assert any("SHARED_DEVICE" in r for r in res.reason_codes)

def test_citation_checker():
    checker = CitationChecker()
    subgraph = {
        "nodes": [{"id": "acc_101", "type": "account"}, {"id": "acc_202", "type": "account"}],
        "edges": [{"event_id": "evt_999", "key": "evt_999"}]
    }

    # Valid statement citing real nodes
    valid_text = "Account [acc_101] sent funds to [acc_202] in event [evt_999]."
    res = checker.verify_explanation(valid_text, subgraph)
    assert res.is_verified is True
    assert "acc_101" in res.cited_nodes
    assert "evt_999" in res.cited_edges

    # Hallucinated statement citing unknown entity
    hallucinated_text = "Account [acc_101] sent funds to phantom account [acc_GHOST]."
    res_bad = checker.verify_explanation(hallucinated_text, subgraph)
    assert res_bad.is_verified is False
    assert "acc_GHOST" not in res_bad.cited_nodes

def test_three_way_calibrator_abstain():
    # Low apparent risk (p_gnn=0.20), but high missingness (M=0.60)
    res = calibrator.calibrate(
        p_tab=0.25,
        p_cnn=0.20,
        p_gnn=0.20,
        m=0.60,
        reason_codes=[]
    )
    # Must abstain as INSUFFICIENT_VISIBILITY instead of blindly ALLOWing!
    assert res["decision"] == DecisionType.INSUFFICIENT_VISIBILITY
    assert "INSUFFICIENT_NETWORK_VISIBILITY" in res["reason_codes"]

def test_pipeline_end_to_end():
    ev = TransactionEvent(
        source_account="acc_pipeline_test_src",
        target_account="acc_pipeline_test_tgt",
        amount=500.0,
        currency="USD"
    )
    res = engine_pipeline.process_event(ev, run_layer2_sync=True, dispatch_slack=False)
    assert "case_id" in res
    assert "decision" in res
    assert "p_tabular" in res
    assert "grounded_summary" in res
    assert res["is_grounding_verified"] is True
