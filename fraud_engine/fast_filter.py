import time
from typing import List
from .types import TransactionEvent, Layer1Result, DecisionType
from .graph_builder import global_graph

class FastFilter:
    """
    Layer 1 Synchronous Filter (Target: < 25ms)
    Executes instant deterministic heuristics before triggering deep Layer 2 models:
    - Shared device count (> 2 accounts on 1 device within short window)
    - Rapid refund loop (< 90 seconds after incoming transfer)
    - High-velocity spike (> 5 transactions within 60 seconds)
    """

    def __init__(self):
        pass

    def evaluate(self, event: TransactionEvent) -> Layer1Result:
        start_time = time.perf_counter()
        reason_codes: List[str] = []
        is_immediate_block = False

        # 1. Shared Device Check
        if event.device_id:
            shared_accounts = global_graph.get_accounts_sharing_device(event.device_id)
            # If current source account is new to this device, count would increase
            effective_count = len(set(shared_accounts + [event.source_account]))
            if effective_count >= 3:
                reason_codes.append(f"SHARED_DEVICE_{effective_count}_ACCOUNTS")
                is_immediate_block = True
            elif effective_count == 2:
                reason_codes.append("SHARED_DEVICE_2_ACCOUNTS")

        # 2. Rapid Refund / Cycle Heuristic
        recent_txs = global_graph.get_recent_transactions(event.source_account, window_seconds=120.0)
        recent_incoming = [tx for tx in recent_txs if tx.get("direction") == "IN"]
        if recent_incoming and event.amount > 0:
            latest_in = recent_incoming[0]
            time_delta = abs(event.timestamp - latest_in.get("timestamp", event.timestamp))
            if time_delta < 90.0:
                reason_codes.append("REFUND_OR_OUTFLOW_WITHIN_90S")
                if time_delta < 30.0 and abs(event.amount - latest_in.get("amount", 0.0)) < 5.0:
                    reason_codes.append("EXACT_RAPID_PASS_THROUGH")
                    is_immediate_block = True

        # 3. Burst Velocity
        recent_outgoing = [tx for tx in recent_txs if tx.get("direction") == "OUT"]
        if len(recent_outgoing) >= 4:
            reason_codes.append("HIGH_VELOCITY_BURST_OUTFLOW")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision = None
        if is_immediate_block:
            decision = DecisionType.BLOCK

        return Layer1Result(
            event_id=event.event_id,
            fast_decision=decision,
            is_immediate_block=is_immediate_block,
            reason_codes=reason_codes,
            latency_ms=round(latency_ms, 2)
        )

fast_filter = FastFilter()
