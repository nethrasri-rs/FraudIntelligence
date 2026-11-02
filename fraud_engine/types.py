from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import uuid

class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    INSUFFICIENT_VISIBILITY = "INSUFFICIENT_VISIBILITY"
    REVIEW = "REVIEW"

class EventType(str, Enum):
    PAYMENT = "PAYMENT"
    REFUND = "REFUND"
    TRANSFER = "TRANSFER"
    PAYPAL_CAPTURE = "PAYPAL_CAPTURE"
    PAYPAL_ORDER = "PAYPAL_ORDER"

class TransactionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    timestamp: float = Field(default_factory=lambda: time.time())
    source_account: str
    target_account: str
    amount: float
    currency: str = "USD"
    event_type: EventType = EventType.PAYMENT
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    source_institution: str = "PAYPAL_HOME_BANK"
    target_institution: str = "PAYPAL_HOME_BANK"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Layer1Result(BaseModel):
    event_id: str
    fast_decision: Optional[DecisionType] = None
    is_immediate_block: bool = False
    reason_codes: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

class SubgraphData(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    center_node: str

class GroundedSummary(BaseModel):
    summary_text: str
    cited_nodes: List[str]
    cited_edges: List[str]
    is_verified: bool
    rejection_reason: Optional[str] = None

class FraudCase(BaseModel):
    case_id: str = Field(default_factory=lambda: f"case_{uuid.uuid4().hex[:8]}")
    event_id: str
    created_at: float = Field(default_factory=lambda: time.time())
    account_id: str
    target_account: Optional[str] = None
    amount: float
    currency: str = "USD"
    p_tabular: float = 0.0
    p_cnn: float = 0.0
    p_gnn: float = 0.0
    p_hybrid: float = 0.0
    visibility_m: float = 0.0
    decision: DecisionType = DecisionType.REVIEW
    reason_codes: List[str] = Field(default_factory=list)
    grounded_summary: Optional[str] = None
    subgraph: Optional[Dict[str, Any]] = None
    analyst_action: Optional[str] = None
    analyst_timestamp: Optional[float] = None
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
