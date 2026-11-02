import hmac
import hashlib
import time
import json
import requests
from typing import Dict, Any, Optional
from fraud_engine.config import config

class WebhookDispatcher:
    """
    Priority 1 Headless Delivery: Signed JSON Webhook Dispatcher.
    Delivers action notifications (e.g. HOLD_FUNDS, FREEZE_ACCOUNT, RISK_UPDATE)
    to downstream bank cores with HMAC-SHA256 signatures, timestamps, and idempotency keys.
    """

    def __init__(self, secret: str = None):
        self.secret = (secret or config.WEBHOOK_SECRET).encode('utf-8')

    def generate_signature(self, timestamp: int, payload_bytes: bytes) -> str:
        msg = f"{timestamp}.".encode('utf-8') + payload_bytes
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()

    def dispatch(
        self,
        endpoint_url: str,
        event_name: str,
        data: Dict[str, Any],
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        timestamp = int(time.time())
        idempotency_key = idempotency_key or f"fi_idem_{timestamp}_{hash(str(data)) % 100000}"

        payload = {
            "event": event_name,
            "timestamp": timestamp,
            "idempotency_key": idempotency_key,
            "data": data
        }
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = self.generate_signature(timestamp, payload_bytes)

        headers = {
            "Content-Type": "application/json",
            "X-FraudIntel-Signature": f"t={timestamp},v1={signature}",
            "X-FraudIntel-Idempotency": idempotency_key
        }

        try:
            resp = requests.post(endpoint_url, data=payload_bytes, headers=headers, timeout=5)
            return {
                "delivered": True,
                "status_code": resp.status_code,
                "idempotency_key": idempotency_key,
                "endpoint": endpoint_url
            }
        except Exception as e:
            # Return logged attempt for headless audit log
            return {
                "delivered": False,
                "error": str(e),
                "idempotency_key": idempotency_key,
                "endpoint": endpoint_url
            }

webhook_dispatcher = WebhookDispatcher()
