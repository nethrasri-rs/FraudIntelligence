from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time

from fraud_engine.types import TransactionEvent, FraudCase, DecisionType
from fraud_engine.pipeline import engine_pipeline
from fraud_engine.graph_builder import global_graph
from integrations.paypal import paypal_client
from storage.db import db
from data.synthetic_aml import aml_generator
from benchmark.edge_drop_eval import benchmark_runner

app = FastAPI(
    title="Fraud Intelligence Engine",
    description="API-first Missingness-Aware Graph Fraud Engine for PayPal Ecosystem",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the AG Grid enterprise investigation console."""
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>UI index.html not found</h3>")

@app.post("/v1/events")
async def ingest_event(event: TransactionEvent):
    """
    Primary headless event intake:
    Executes Layer 1 synchronously (<25ms) and triggers Layer 2 deep graph inference.
    """
    result = engine_pipeline.process_event(event, run_layer2_sync=True, dispatch_slack=True)
    return result

@app.post("/v1/webhooks/paypal")
async def paypal_webhook_intake(request: Request):
    """
    Direct ingestion for PayPal Developer Sandbox Webhooks.
    Normalizes payloads (CHECKOUT.ORDER.APPROVED, PAYMENT.CAPTURE.COMPLETED, REFUNDS)
    and scores transactions in real time.
    """
    payload = await request.json()
    device_header = request.headers.get("X-PayPal-Device-Id")
    event = paypal_client.normalize_webhook(payload, device_id=device_header)
    result = engine_pipeline.process_event(event, run_layer2_sync=True, dispatch_slack=True)
    return {
        "status": "ACCEPTED",
        "paypal_event_id": payload.get("id"),
        "engine_result": result
    }

@app.get("/v1/cases")
async def list_cases(limit: int = 100):
    """Returns cases for the AG Grid console."""
    return db.list_cases(limit=limit)

@app.get("/v1/cases/{case_id}")
async def get_case_details(case_id: str):
    c = db.get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="Case not found")
    c["audit_trail"] = db.get_audit_trail(case_id)
    return c

@app.post("/v1/cases/{case_id}/action")
async def submit_case_action(case_id: str, payload: dict):
    """Records analyst human-in-the-loop action into the immutable audit log."""
    action = payload.get("action", "REVIEW")
    actor = payload.get("actor", "analyst_console")
    db.record_analyst_action(case_id, action, actor)
    return {"status": "SUCCESS", "case_id": case_id, "action": action}

@app.post("/v1/simulate/replay")
async def trigger_replay():
    """Replays the synthetic AML stream with planted mule rings and cycles (Demo A)."""
    stream = aml_generator.generate_benchmark_stream(total_transactions=25)
    processed = []
    for ev in stream:
        res = engine_pipeline.process_event(ev, run_layer2_sync=True, dispatch_slack=False)
        processed.append(res)
    return {"status": "COMPLETED", "events_replayed": len(processed)}

@app.get("/v1/simulate/edge-drop")
async def run_edge_drop_sim():
    """Executes the edge-drop robustness benchmark (Demo B)."""
    results = benchmark_runner.run_benchmark(drop_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    return results

@app.post("/v1/system/reset")
async def system_reset():
    """Resets in-memory graph and database for reproducible demo presentation."""
    global_graph.clear()
    db.reset_database()
    return {"status": "RESET_SUCCESSFUL"}

@app.post("/v1/mock/bank-core/hold")
async def mock_bank_core_hold(request: Request):
    """Mock downstream bank core webhook receiver for fund freeze instructions."""
    data = await request.json()
    return {"status": "HOLD_APPLIED", "received_at": time.time(), "details": data}
