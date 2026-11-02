import random
import time
import uuid
from typing import List, Dict, Any
from fraud_engine.types import TransactionEvent, EventType
from fraud_engine.config import config

class SyntheticAMLGenerator:
    """
    Generates realistic synthetic AML & fraud transaction streams
    with planted laundering motifs:
    1. Fan-out Smurfing (one originator -> 6 mules)
    2. Shared Device Mule Ring (mules transacting from common device fingerprint)
    3. Rapid Refund / Pass-through Loop
    4. Cross-Bank Blindspot Partition (high Missingness M)
    5. Clean Benign Commerce
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.home_inst = config.HOME_INSTITUTION_ID
        self.ext_inst = "EXTERNAL_BANK_CHASE"

    def generate_benchmark_stream(self, total_transactions: int = 120) -> List[TransactionEvent]:
        events: List[TransactionEvent] = []
        base_time = time.time() - (total_transactions * 15)

        # 1. Planted Shared-Device Mule Ring
        originator = "acc_mule_boss_901"
        mule_accounts = [f"acc_mule_inter_{i}" for i in range(1, 7)]
        shared_device_a = "dev_fingerprint_mule_cluster_99"
        shared_device_b = "dev_fingerprint_mule_cluster_99" # shared device!

        # Originator fans out to 6 intermediaries within 2 minutes
        for i, mule in enumerate(mule_accounts):
            t = base_time + (i * 12)
            dev = shared_device_a if i < 3 else f"dev_clean_{i}"
            events.append(TransactionEvent(
                event_id=f"evt_fanout_{i+1}",
                timestamp=t,
                source_account=originator,
                target_account=mule,
                amount=round(950.0 + random.uniform(5.0, 45.0), 2),
                currency="USD",
                event_type=EventType.TRANSFER,
                device_id=dev,
                source_institution=self.home_inst,
                target_institution=self.home_inst,
                metadata={"pattern": "fan_out_mule", "label": "FRAUD"}
            ))

        # Two mules cycle funds back or forward on the shared device (cycle motif)
        events.append(TransactionEvent(
            event_id="evt_cycle_pass_01",
            timestamp=base_time + 100,
            source_account=mule_accounts[0],
            target_account=mule_accounts[1],
            amount=880.0,
            currency="USD",
            event_type=EventType.TRANSFER,
            device_id=shared_device_a,
            source_institution=self.home_inst,
            target_institution=self.home_inst,
            metadata={"pattern": "cycle_link", "label": "FRAUD"}
        ))

        # 2. Planted Rapid Refund Pass-through
        victim_payer = "acc_victim_merchant_55"
        fraudster_target = "acc_refund_smurf_88"
        refund_dev = "dev_refund_bot_12"

        events.append(TransactionEvent(
            event_id="evt_refund_inflow_01",
            timestamp=base_time + 150,
            source_account=victim_payer,
            target_account=fraudster_target,
            amount=1450.0,
            currency="USD",
            event_type=EventType.PAYMENT,
            device_id=refund_dev,
            source_institution=self.home_inst,
            target_institution=self.home_inst,
            metadata={"pattern": "rapid_refund_inflow", "label": "LEGIT"}
        ))
        # 45 seconds later: rapid refund/pass-through
        events.append(TransactionEvent(
            event_id="evt_refund_drain_02",
            timestamp=base_time + 195,
            source_account=fraudster_target,
            target_account="acc_cashout_crypto_gateway",
            amount=1445.0,
            currency="USD",
            event_type=EventType.REFUND,
            device_id=refund_dev,
            source_institution=self.home_inst,
            target_institution=self.home_inst,
            metadata={"pattern": "rapid_refund_drain", "label": "FRAUD"}
        ))

        # 3. Planted Cross-Bank Blindspot Partition (High Missingness M)
        # Account interacts heavily with EXTERNAL bank, invisible to internal GraphSAGE
        blindspot_account = "acc_crossbank_hidden_77"
        for j in range(5):
            events.append(TransactionEvent(
                event_id=f"evt_crossbank_{j+1}",
                timestamp=base_time + 250 + (j * 20),
                source_account=blindspot_account,
                target_account=f"acc_ext_chase_{j+1}",
                amount=round(2400.0 + random.uniform(10.0, 100.0), 2),
                currency="USD",
                event_type=EventType.TRANSFER,
                device_id=f"dev_cross_{j}",
                source_institution=self.home_inst,
                target_institution=self.ext_inst, # EXTERNAL!
                metadata={"pattern": "cross_bank_blindspot", "label": "INSUFFICIENT_VISIBILITY"}
            ))

        # 4. Background Benign Commercial Traffic
        clean_users = [f"acc_clean_customer_{k}" for k in range(1, 20)]
        clean_merchants = [f"acc_merchant_store_{k}" for k in range(1, 6)]

        for k in range(total_transactions - len(events)):
            t = base_time + 350 + (k * 15)
            u = random.choice(clean_users)
            m = random.choice(clean_merchants)
            amt = round(random.uniform(12.0, 240.0), 2)
            events.append(TransactionEvent(
                event_id=f"evt_clean_{k+1}",
                timestamp=t,
                source_account=u,
                target_account=m,
                amount=amt,
                currency="USD",
                event_type=EventType.PAYMENT,
                device_id=f"dev_phone_{u}",
                source_institution=self.home_inst,
                target_institution=self.home_inst,
                metadata={"pattern": "benign_commerce", "label": "LEGIT"}
            ))

        # Chronological sort
        events.sort(key=lambda x: x.timestamp)
        return events

aml_generator = SyntheticAMLGenerator()
