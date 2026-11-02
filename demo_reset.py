"""
Demo Reset Utility
Resets the database, graph state, and pre-populates a fresh slice of transactions.
"""
from fraud_engine.graph_builder import global_graph
from storage.db import db
from data.synthetic_aml import aml_generator
from fraud_engine.pipeline import engine_pipeline

def reset_all(seed: int = 42):
    print("Resetting Fraud Intelligence Engine state...")
    global_graph.clear()
    db.reset_database()
    print("Database and graph cleared.")

    print("Replaying initial transaction baseline stream...")
    events = aml_generator.generate_benchmark_stream(total_transactions=20)
    for ev in events:
        engine_pipeline.process_event(ev, run_layer2_sync=True, dispatch_slack=False)

    print("Demo reset complete! Ready for live testing and presentation.")

if __name__ == "__main__":
    reset_all()
