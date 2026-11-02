import requests
import json
from typing import Dict, Any, List, Optional
from fraud_engine.config import config
from fraud_engine.types import FraudCase, DecisionType

class SlackAlertDispatcher:
    """
    Constructs high-priority Slack Block Kit alert cards
    with interactive action buttons and delivers them to Slack channels.
    """

    def __init__(self):
        self.webhook_url = config.SLACK_WEBHOOK_URL

    def build_case_block_kit(self, case: FraudCase) -> Dict[str, Any]:
        """
        Creates Block Kit payload matching Section 7 of Blueprint:
        - Risk header with P_hybrid and Missingness M
        - Decision badge (BLOCK / ALLOW / INSUFFICIENT VISIBILITY)
        - Reason codes list
        - Grounded summary with cited tokens
        - Interactive action buttons: [Approve], [Block], [Escalate Cross-Bank]
        """
        # Determine color emoji / status banner
        if case.decision == DecisionType.BLOCK:
            status_text = "🚨 *FRAUD ALERT: HIGH RISK DETECTED*"
            decision_badge = "🔴 `BLOCK` (Immediate Action Recommended)"
        elif case.decision == DecisionType.INSUFFICIENT_VISIBILITY:
            status_text = "⚠️ *NETWORK BLIND-SPOT: INSUFFICIENT VISIBILITY*"
            decision_badge = "🟡 `INSUFFICIENT VISIBILITY` (Abstain & Escalate)"
        elif case.decision == DecisionType.ALLOW:
            status_text = "✅ *CLEAN TRANSACTION: LOW RISK*"
            decision_badge = "🟢 `ALLOW`"
        else:
            status_text = "🔍 *CASE FLAGGED FOR ANALYST REVIEW*"
            decision_badge = "🔵 `REVIEW`"

        reasons_formatted = "\n".join([f"• `{r}`" for r in case.reason_codes]) if case.reason_codes else "None"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Fraud Intelligence Alert — Case {case.case_id[:12]}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{status_text}\n*Decision:* {decision_badge}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Account:*\n`{case.account_id}`"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Amount:*\n${case.amount:.2f} {case.currency}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Hybrid Risk (P_hybrid):*\n`{case.p_hybrid:.3f}`"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Missingness Signal (M):*\n`{case.visibility_m:.3f}`"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Model Breakdown:*\n• Tabular: `{case.p_tabular:.3f}` | Causal CNN: `{case.p_cnn:.3f}` | GraphSAGE: `{case.p_gnn:.3f}`"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Triggered Signals / Reason Codes:*\n{reasons_formatted}"
                }
            }
        ]

        # Add Grounded Explanation
        if case.grounded_summary:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Grounded AI Evidence Summary:*\n>{case.grounded_summary}"
                }
            })

        # Add Interactive Buttons
        blocks.append({
            "type": "actions",
            "block_id": f"action_block_{case.case_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Approve",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": f"approve_{case.case_id}",
                    "action_id": "btn_approve"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Block Account",
                        "emoji": True
                    },
                    "style": "danger",
                    "value": f"block_{case.case_id}",
                    "action_id": "btn_block"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Escalate (Cross-Bank)",
                        "emoji": True
                    },
                    "value": f"escalate_{case.case_id}",
                    "action_id": "btn_escalate"
                }
            ]
        })

        return {"blocks": blocks}

    def send_alert(self, case: FraudCase) -> bool:
        """Sends the Block Kit payload to the configured Slack Webhook URL."""
        if not self.webhook_url:
            print("[Slack] No webhook URL configured, skipping Slack dispatch.")
            return False

        payload = self.build_case_block_kit(case)
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if resp.status_code == 200:
                print(f"[Slack] Successfully dispatched Block Kit alert for case {case.case_id}")
                return True
            else:
                print(f"[Slack] Webhook returned status {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            print(f"[Slack Dispatch Error]: {e}")
            return False

slack_dispatcher = SlackAlertDispatcher()
