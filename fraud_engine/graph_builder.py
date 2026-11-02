import networkx as nx
import time
from typing import Dict, Any, List, Optional
from .types import TransactionEvent, SubgraphData

class TransactionGraph:
    """
    Maintains a real-time heterogeneous payment & entity network graph:
    - Nodes:
        type='account' (account_id, institution, balance, creation_time)
        type='device'  (device_id, ip)
    - Edges:
        type='transaction' (amount, timestamp, currency, event_id)
        type='used_device' (last_seen)
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.events_history: List[TransactionEvent] = []

    def clear(self):
        self.G.clear()
        self.events_history.clear()

    def add_transaction(self, event: TransactionEvent):
        self.events_history.append(event)

        # Add Source Account Node
        if not self.G.has_node(event.source_account):
            self.G.add_node(
                event.source_account,
                node_type="account",
                institution=event.source_institution,
                first_seen=event.timestamp,
                tx_count=0
            )
        self.G.nodes[event.source_account]["tx_count"] = (
            self.G.nodes[event.source_account].get("tx_count", 0) + 1
        )

        # Add Target Account Node
        if not self.G.has_node(event.target_account):
            self.G.add_node(
                event.target_account,
                node_type="account",
                institution=event.target_institution,
                first_seen=event.timestamp,
                tx_count=0
            )
        self.G.nodes[event.target_account]["tx_count"] = (
            self.G.nodes[event.target_account].get("tx_count", 0) + 1
        )

        # Add Transaction Directed Edge
        self.G.add_edge(
            event.source_account,
            event.target_account,
            key=event.event_id,
            edge_type="transaction",
            event_id=event.event_id,
            amount=event.amount,
            currency=event.currency,
            timestamp=event.timestamp,
            event_type=str(event.event_type)
        )

        # Device connection if present
        if event.device_id:
            dev_node = f"dev_{event.device_id}"
            if not self.G.has_node(dev_node):
                self.G.add_node(
                    dev_node,
                    node_type="device",
                    device_id=event.device_id,
                    first_seen=event.timestamp
                )

            # Link account to device
            self.G.add_edge(
                event.source_account,
                dev_node,
                key=f"dev_link_{event.event_id}",
                edge_type="used_device",
                timestamp=event.timestamp
            )

    def get_accounts_sharing_device(self, device_id: str) -> List[str]:
        """Finds all distinct accounts that connected to this device."""
        dev_node = f"dev_{device_id}"
        if not self.G.has_node(dev_node):
            return []
        # Predecessors of device node
        accounts = [u for u, v, data in self.G.in_edges(dev_node, data=True) if data.get("edge_type") == "used_device"]
        return list(set(accounts))

    def get_recent_transactions(self, account_id: str, window_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """Returns recent outgoing and incoming transactions within a time window."""
        now = time.time()
        recent = []
        if not self.G.has_node(account_id):
            return recent

        # Outgoing
        for _, v, data in self.G.out_edges(account_id, data=True):
            if data.get("edge_type") == "transaction":
                if (now - data.get("timestamp", 0)) <= window_seconds:
                    recent.append({**data, "direction": "OUT", "counterparty": v})

        # Incoming
        for u, _, data in self.G.in_edges(account_id, data=True):
            if data.get("edge_type") == "transaction":
                if (now - data.get("timestamp", 0)) <= window_seconds:
                    recent.append({**data, "direction": "IN", "counterparty": u})

        recent.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return recent

    def extract_subgraph(self, center_node: str, depth: int = 2) -> SubgraphData:
        """
        Extracts k-hop ego subgraph centered on center_node
        for visualization and GNN inference.
        """
        if center_node not in self.G:
            return SubgraphData(nodes=[], edges=[], center_node=center_node)

        # Convert to undirected view for k-hop neighborhood search
        undirected_view = self.G.to_undirected(as_view=True)
        try:
            sub_nodes = set(nx.single_source_shortest_path_length(undirected_view, center_node, cutoff=depth).keys())
        except Exception:
            sub_nodes = {center_node}

        nodes_list = []
        for n in sub_nodes:
            attrs = self.G.nodes[n]
            nodes_list.append({
                "id": str(n),
                "type": attrs.get("node_type", "account"),
                "institution": attrs.get("institution", "PAYPAL_HOME_BANK"),
                "tx_count": attrs.get("tx_count", 0),
                "is_center": (n == center_node)
            })

        edges_list = []
        for u in sub_nodes:
            for v in self.G[u]:
                if v in sub_nodes:
                    for k, edge_data in self.G[u][v].items():
                        edges_list.append({
                            "source": str(u),
                            "target": str(v),
                            "key": str(k),
                            "type": edge_data.get("edge_type", "transaction"),
                            "amount": edge_data.get("amount", 0.0),
                            "event_id": edge_data.get("event_id", str(k)),
                            "timestamp": edge_data.get("timestamp", 0.0)
                        })

        return SubgraphData(
            nodes=nodes_list,
            edges=edges_list,
            center_node=center_node
        )

# Global in-memory graph
global_graph = TransactionGraph()
