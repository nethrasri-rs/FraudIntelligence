import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from fraud_engine.config import config
from fraud_engine.types import FraudCase, DecisionType

DB_PATH = config.DATABASE_PATH

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Cases table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                account_id TEXT NOT NULL,
                target_account TEXT,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                p_tabular REAL NOT NULL,
                p_cnn REAL NOT NULL,
                p_gnn REAL NOT NULL,
                p_hybrid REAL NOT NULL,
                visibility_m REAL NOT NULL,
                decision TEXT NOT NULL,
                reason_codes TEXT NOT NULL,
                grounded_summary TEXT,
                subgraph_json TEXT,
                analyst_action TEXT,
                analyst_timestamp REAL
            )
            """)

            # Audit Trail table (immutable events)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                details TEXT NOT NULL
            )
            """)

            # Raw Ingested Events table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                source_account TEXT NOT NULL,
                target_account TEXT NOT NULL,
                amount REAL NOT NULL,
                event_type TEXT NOT NULL,
                device_id TEXT,
                payload_json TEXT NOT NULL
            )
            """)
            conn.commit()

    def save_event(self, event_dict: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO events (
                event_id, timestamp, source_account, target_account, amount, event_type, device_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_dict["event_id"],
                event_dict.get("timestamp", time.time()),
                event_dict["source_account"],
                event_dict["target_account"],
                event_dict["amount"],
                str(event_dict.get("event_type", "PAYMENT")),
                event_dict.get("device_id"),
                json.dumps(event_dict)
            ))
            conn.commit()

    def save_case(self, case: FraudCase):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO cases (
                case_id, event_id, created_at, account_id, target_account,
                amount, currency, p_tabular, p_cnn, p_gnn, p_hybrid,
                visibility_m, decision, reason_codes, grounded_summary,
                subgraph_json, analyst_action, analyst_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case.case_id,
                case.event_id,
                case.created_at,
                case.account_id,
                case.target_account,
                case.amount,
                case.currency,
                case.p_tabular,
                case.p_cnn,
                case.p_gnn,
                case.p_hybrid,
                case.visibility_m,
                case.decision.value if hasattr(case.decision, "value") else str(case.decision),
                json.dumps(case.reason_codes),
                case.grounded_summary,
                json.dumps(case.subgraph) if case.subgraph else None,
                case.analyst_action,
                case.analyst_timestamp
            ))

            # Record creation in audit trail
            cursor.execute("""
            INSERT INTO audit_logs (case_id, timestamp, action, actor, details)
            VALUES (?, ?, ?, ?, ?)
            """, (
                case.case_id,
                time.time(),
                "CASE_CREATED",
                "FRAUD_ENGINE",
                json.dumps({
                    "decision": str(case.decision),
                    "p_hybrid": case.p_hybrid,
                    "m_score": case.visibility_m,
                    "reasons": case.reason_codes
                })
            ))
            conn.commit()

    def record_analyst_action(self, case_id: str, action: str, actor: str = "analyst_web"):
        now = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE cases
            SET analyst_action = ?, analyst_timestamp = ?
            WHERE case_id = ?
            """, (action, now, case_id))

            cursor.execute("""
            INSERT INTO audit_logs (case_id, timestamp, action, actor, details)
            VALUES (?, ?, ?, ?, ?)
            """, (case_id, now, f"ANALYST_{action.upper()}", actor, json.dumps({"action": action})))
            conn.commit()

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["reason_codes"] = json.loads(d["reason_codes"]) if d["reason_codes"] else []
            d["subgraph"] = json.loads(d["subgraph_json"]) if d["subgraph_json"] else None
            return d

    def list_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["reason_codes"] = json.loads(d["reason_codes"]) if d["reason_codes"] else []
                d["subgraph"] = json.loads(d["subgraph_json"]) if d["subgraph_json"] else None
                results.append(d)
            return results

    def get_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs WHERE case_id = ? ORDER BY timestamp ASC", (case_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def reset_database(self):
        """Clean wipe for reproducible demo resets."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cases")
            cursor.execute("DELETE FROM audit_logs")
            cursor.execute("DELETE FROM events")
            conn.commit()

db = Database()
