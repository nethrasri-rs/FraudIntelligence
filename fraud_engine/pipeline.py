import time
from typing import Dict, Any, Optional
from fraud_engine.types import TransactionEvent, Layer1Result, FraudCase, DecisionType
from fraud_engine.graph_builder import global_graph
from fraud_engine.fast_filter import fast_filter
from fraud_engine.visibility import visibility_estimator
from fraud_engine.models.tabular import tabular_model
from fraud_engine.models.causal_cnn import causal_cnn
from fraud_engine.models.gnn_missingness import gnn_model
from fraud_engine.models.fusion import calibrator
from fraud_engine.explainer import grounded_explainer
from storage.db import db
from integrations.slack import slack_dispatcher
from integrations.dispatcher import webhook_dispatcher

class FraudEnginePipeline:
    """
    End-to-end Fraud Intelligence Engine Pipeline:
    Coordinates Layer 1 (<25ms fast filter) and Layer 2 (50-120ms deep inference).
    """

    def process_event(
        self,
        event: TransactionEvent,
        run_layer2_sync: bool = True,
        dispatch_slack: bool = True
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Ingest into raw event store
        db.save_event(event.model_dump())

        # Update in-memory graph
        global_graph.add_transaction(event)

        # 1. LAYER 1: Synchronous Fast Filter (< 25ms)
        l1_result = fast_filter.evaluate(event)

        # Base case response structure
        response = {
            "event_id": event.event_id,
            "decision": l1_result.fast_decision.value if l1_result.fast_decision else "REVIEW",
            "risk": 0.85 if l1_result.is_immediate_block else 0.40,
            "visibility_m": 0.0,
            "reason_codes": l1_result.reason_codes,
            "layer1_latency_ms": l1_result.latency_ms,
            "layer2_pending": not run_layer2_sync
        }

        if not run_layer2_sync:
            return response

        # 2. LAYER 2: Deep Missingness-Aware Evaluation
        t_l2_start = time.perf_counter()

        # Compute Visibility / Missingness M
        m_info = visibility_estimator.compute_node_missingness(
            global_graph.G,
            event.source_account
        )
        visibility_m = m_info["m_score"]

        # Branch Scores
        p_tab = tabular_model.predict_proba(event)
        p_cnn = causal_cnn.predict_proba(event)
        gnn_res = gnn_model.predict_proba(event)
        p_gnn = gnn_res["p_gnn"]

        # Calibrate & Fuse
        all_reasons = list(set(l1_result.reason_codes))
        if gnn_res["patterns"]["cycle"] > 0.5:
            all_reasons.append("DIRECTED_MULE_CYCLE")
        if gnn_res["patterns"]["fan_out"] > 0.7:
            all_reasons.append("HIGH_FAN_OUT_SMURFING")
        if gnn_res["patterns"]["fan_in"] > 0.7:
            all_reasons.append("HIGH_FAN_IN_AGGREGATION")

        cal_res = calibrator.calibrate(p_tab, p_cnn, p_gnn, visibility_m, all_reasons)
        p_hybrid = cal_res["p_hybrid"]
        decision = cal_res["decision"]

        # If Layer 1 flagged immediate block, maintain BLOCK decision
        if l1_result.is_immediate_block:
            decision = DecisionType.BLOCK

        # Extract Ego Subgraph
        subgraph_data = global_graph.extract_subgraph(event.source_account, depth=2)
        subgraph_dict = subgraph_data.model_dump()

        # Grounded AI Case Summary with strict citation checker
        grounded = grounded_explainer.generate_explanation(
            account_id=event.source_account,
            reason_codes=cal_res["reason_codes"],
            p_hybrid=p_hybrid,
            visibility_m=visibility_m,
            subgraph=subgraph_dict
        )

        l2_latency_ms = (time.perf_counter() - t_l2_start) * 1000.0

        # Create and Save Case
        case = FraudCase(
            event_id=event.event_id,
            account_id=event.source_account,
            target_account=event.target_account,
            amount=event.amount,
            currency=event.currency,
            p_tabular=p_tab,
            p_cnn=p_cnn,
            p_gnn=p_gnn,
            p_hybrid=p_hybrid,
            visibility_m=visibility_m,
            decision=decision,
            reason_codes=cal_res["reason_codes"],
            grounded_summary=grounded.summary_text,
            subgraph=subgraph_dict
        )
        db.save_case(case)

        # 3. Headless Delivery Channels
        if dispatch_slack and decision in [DecisionType.BLOCK, DecisionType.INSUFFICIENT_VISIBILITY, DecisionType.REVIEW]:
            slack_dispatcher.send_alert(case)

        # Webhook Notification for Bank Core Holds
        if decision == DecisionType.BLOCK:
            webhook_dispatcher.dispatch(
                endpoint_url="http://localhost:8000/v1/mock/bank-core/hold",
                event_name="PAYMENT_HOLD_REQUESTED",
                data={
                    "case_id": case.case_id,
                    "account_id": case.account_id,
                    "event_id": event.event_id,
                    "action": "IMMEDIATE_HOLD",
                    "reason": cal_res["reason_codes"]
                }
            )

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "case_id": case.case_id,
            "event_id": event.event_id,
            "decision": decision.value if hasattr(decision, "value") else str(decision),
            "risk": p_hybrid,
            "visibility_m": visibility_m,
            "p_tabular": p_tab,
            "p_cnn": p_cnn,
            "p_gnn": p_gnn,
            "reason_codes": cal_res["reason_codes"],
            "grounded_summary": grounded.summary_text,
            "is_grounding_verified": grounded.is_verified,
            "layer1_latency_ms": l1_result.latency_ms,
            "layer2_latency_ms": round(l2_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "subgraph": subgraph_dict
        }

engine_pipeline = FraudEnginePipeline()
