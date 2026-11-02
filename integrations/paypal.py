import requests
import json
import base64
import time
from typing import Dict, Any, Optional
from fraud_engine.config import config
from fraud_engine.types import TransactionEvent, EventType

class PayPalSandboxClient:
    """
    Direct integration with PayPal Developer Sandbox APIs:
    - OAuth2 Authentication (Client Credentials Grant)
    - Orders API v2 (Create & Capture Orders)
    - Payments API v2 (Refunds)
    - Webhook Event Normalization for the Fraud Engine
    """

    def __init__(self):
        self.base_url = config.PAYPAL_BASE_URL.rstrip('/')
        self.client_id = config.PAYPAL_CLIENT_ID
        self.client_secret = config.PAYPAL_CLIENT_SECRET
        self._cached_token = None
        self._token_expires_at = 0

    def get_access_token(self) -> str:
        """Fetches OAuth2 bearer token from PayPal Sandbox."""
        if self._cached_token and time.time() < (self._token_expires_at - 60):
            return self._cached_token

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        try:
            resp = requests.post(
                f"{self.base_url}/v1/oauth2/token",
                headers=headers,
                data=data,
                timeout=10
            )
            resp.raise_for_status()
            payload = resp.json()
            self._cached_token = payload["access_token"]
            self._token_expires_at = time.time() + payload.get("expires_in", 3600)
            return self._cached_token
        except Exception as e:
            # Fallback for offline sandbox mode
            print(f"[PayPal Sandbox Warning] OAuth2 request failed ({e}). Using mock session.")
            self._cached_token = "SANDBOX_MOCK_TOKEN_" + hex(int(time.time()))
            self._token_expires_at = time.time() + 3600
            return self._cached_token

    def create_sandbox_order(
        self,
        amount: float,
        currency: str = "USD",
        custom_id: str = "fi_order_001"
    ) -> Dict[str, Any]:
        """Creates an order in PayPal Sandbox."""
        token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        body = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": custom_id,
                    "amount": {
                        "currency_code": currency,
                        "value": f"{amount:.2f}"
                    },
                    "description": "Fraud Intelligence Verification Order"
                }
            ]
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v2/checkout/orders",
                headers=headers,
                json=body,
                timeout=10
            )
            return resp.json()
        except Exception as e:
            return {
                "id": f"ORD-MOCK-{int(time.time())}",
                "status": "CREATED",
                "simulated": True,
                "error": str(e)
            }

    def normalize_webhook(self, raw_payload: Dict[str, Any], device_id: Optional[str] = None) -> TransactionEvent:
        """
        Translates raw PayPal Sandbox Webhook JSON into unified TransactionEvent.
        Supports:
          - PAYMENT.CAPTURE.COMPLETED
          - PAYMENT.CAPTURE.REFUNDED
          - PAYMENT.CAPTURE.DENIED
          - CHECKOUT.ORDER.APPROVED
        """
        event_type_str = raw_payload.get("event_type", "PAYMENT.CAPTURE.COMPLETED")
        resource = raw_payload.get("resource", {})

        # Extract amount and currency
        amount_obj = resource.get("amount", {})
        amount = float(amount_obj.get("value", 0.0))
        currency = amount_obj.get("currency_code", "USD")

        # Accounts: Payer / Payee
        payer = resource.get("payer", {}).get("payer_id") or resource.get("payer", {}).get("email_address")
        if not payer:
            payer = raw_payload.get("payer_id") or config.PAYPAL_PERSONAL_EMAIL or "paypal_payer_sandbox_01"

        payee = resource.get("payee", {}).get("merchant_id") or resource.get("payee", {}).get("email_address")
        if not payee:
            payee = raw_payload.get("payee_id") or config.PAYPAL_BUSINESS_EMAIL or "paypal_payee_sandbox_02"

        # Determine event type
        if "REFUND" in event_type_str:
            ev_type = EventType.REFUND
            # In refund, money flows back to payer
            source_acc, target_acc = payee, payer
        else:
            ev_type = EventType.PAYPAL_CAPTURE
            source_acc, target_acc = payer, payee

        event_id = raw_payload.get("id") or resource.get("id") or f"paypal_evt_{int(time.time()*1000)}"

        # Custom device token if provided in metadata / headers
        device = device_id or resource.get("custom_id") or "dev_paypal_web_client"

        return TransactionEvent(
            event_id=event_id,
            timestamp=time.time(),
            source_account=source_acc,
            target_account=target_acc,
            amount=amount,
            currency=currency,
            event_type=ev_type,
            device_id=device,
            source_institution=config.HOME_INSTITUTION_ID,
            target_institution=config.HOME_INSTITUTION_ID,
            metadata={"raw_paypal_event": event_type_str, "paypal_resource_id": resource.get("id")}
        )

paypal_client = PayPalSandboxClient()
