from scapy.all import IP, TCP, UDP, ICMP
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'Source IP', 'Destination IP', 'Timestamp',
    'Average Packet Size', 'Bwd Packets/s',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s'
]

class FlowExtractor:

    def __init__(self):
        self.flows = defaultdict(self._create_flow_dict)

    def _create_flow_dict(self):
        return {
            'timestamps': [],
            'packet_sizes': [],
            'bwd_packet_sizes': [],
            'flags': defaultdict(int),
            'forward_src': None,
            'forward_dst': None,
        }

    def process_packet(self, packet):
        if IP not in packet:
            return

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = 'OTHER'

        if TCP in packet:
            protocol = 'TCP'
        elif UDP in packet:
            protocol = 'UDP'
        elif ICMP in packet:
            protocol = 'ICMP'

        ip_pair = tuple(sorted((src_ip, dst_ip)))
        flow_key = (ip_pair[0], ip_pair[1], protocol)
        flow = self.flows[flow_key]

        if flow['forward_src'] is None:
            flow['forward_src'] = src_ip
            flow['forward_dst'] = dst_ip

        timestamp = datetime.fromtimestamp(float(packet.time))
        size = len(packet)

        flow['timestamps'].append(timestamp)
        flow['packet_sizes'].append(size)

        if not (src_ip == flow['forward_src'] and dst_ip == flow['forward_dst']):
            flow['bwd_packet_sizes'].append(size)

        if TCP in packet:
            tcp = packet[TCP]
            if tcp.flags.F: flow['flags']['FIN'] += 1
            if tcp.flags.S: flow['flags']['SYN'] += 1
            if tcp.flags.R: flow['flags']['RST'] += 1
            if tcp.flags.P: flow['flags']['PSH'] += 1
            if tcp.flags.A: flow['flags']['ACK'] += 1
            if tcp.flags.U: flow['flags']['URG'] += 1
            if tcp.flags.E: flow['flags']['ECE'] += 1
            if tcp.flags.C: flow['flags']['CWE'] += 1

    def build_dataframe(self):
        rows = []

        for (_ip_a, _ip_b, protocol), flow in self.flows.items():

            if not flow['timestamps']:
                continue

            start = min(flow['timestamps'])
            end = max(flow['timestamps'])
            duration = (end - start).total_seconds()
            duration = max(duration, 0.001)

            packet_count = len(flow['packet_sizes'])
            avg_packet_size = np.mean(flow['packet_sizes'])
            flow_packets_sec = packet_count / duration
            bwd_packets_sec = len(flow['bwd_packet_sizes']) / duration

            row = {
                'Source IP': flow['forward_src'],
                'Destination IP': flow['forward_dst'],
                'Timestamp': start.isoformat(),
                'Average Packet Size': avg_packet_size,
                'Bwd Packets/s': bwd_packets_sec,
                'FIN Flag Count': flow['flags'].get('FIN', 0),
                'SYN Flag Count': flow['flags'].get('SYN', 0),
                'RST Flag Count': flow['flags'].get('RST', 0),
                'PSH Flag Count': flow['flags'].get('PSH', 0),
                'ACK Flag Count': flow['flags'].get('ACK', 0),
                'URG Flag Count': flow['flags'].get('URG', 0),
                'CWE Flag Count': flow['flags'].get('CWE', 0),
                'ECE Flag Count': flow['flags'].get('ECE', 0),
                'Flow Packets/s': flow_packets_sec,
            }

            rows.append(row)

        return pd.DataFrame(rows, columns=FEATURE_COLS)
