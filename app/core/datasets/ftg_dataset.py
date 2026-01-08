import torch
import pandas as pd
from torch_geometric.data import Data


class FTGDataset:
    def __init__(self, df, time_slot_duration='5s', min_packets_per_flow=1,
                 require_shared_ips=False, min_flows_per_slot=1, has_labels=True):

        self.has_labels = has_labels
        df = df.copy()

        df['Timestamp'] = df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors="coerce")
        df = df.dropna(subset=['Timestamp'])

        if df.empty:
            raise ValueError("No valid timestamps in dataset")

        df = df.set_index('Timestamp').sort_index()
        self.valid_time_slots = [
            g for _, g in df.groupby(pd.Grouper(freq=time_slot_duration)) if not g.empty
        ]

        self.feature_cols = [
            "Average Packet Size", "Bwd Packets/s", "FIN Flag Count",
            "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
            "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
            "ECE Flag Count", "Flow Packets/s"
        ]

    def __len__(self):
        return len(self.valid_time_slots)

    def __getitem__(self, idx):
        slot = self.valid_time_slots[idx]
        endpoint_groups = slot.groupby(["Source IP", "Destination IP"])

        flow_graphs, node_map = [], {}

        for (src_ip, dst_ip), group in endpoint_groups:
            node_map[(src_ip, dst_ip)] = len(node_map)
            x = torch.tensor(group[self.feature_cols].values, dtype=torch.float)

            if len(x) > 1:
                edges = torch.tensor([[i, i + 1] for i in range(len(x) - 1)],
                                     dtype=torch.long).t()
            else:
                edges = torch.empty((2, 0), dtype=torch.long)

            if self.has_labels:
                label = torch.tensor([group["Label"].max()], dtype=torch.float)
            else:
                label = torch.tensor([-1.0], dtype=torch.float)

            flow_graphs.append(Data(x=x, edge_index=edges, y=label))

        traffic_edges = []
        for i in range(len(flow_graphs)):
            for j in range(i + 1, len(flow_graphs)):
                traffic_edges += [[i, j], [j, i]]

        if not traffic_edges:
            traffic_edges = [[0, 0]]

        traffic_graph = Data(
            x=torch.empty((len(flow_graphs), len(self.feature_cols))),
            edge_index=torch.tensor(traffic_edges, dtype=torch.long).t(),
            y=torch.cat([fg.y for fg in flow_graphs])
        )

        return traffic_graph, flow_graphs
    
