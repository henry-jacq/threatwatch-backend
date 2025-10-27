"""
PCAP to CICFlowMeter-compatible Dataset Converter
Converts raw PCAP files to the 11-feature dataset format
"""
import logging
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class FlowExtractor:
    """Extract flows from PCAP with CICFlowMeter-like features"""
    
    # Feature columns matching your model
    FEATURE_COLS = [
        'Source IP', 'Destination IP', 'Timestamp', 'Label',
        'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 
        'ECE Flag Count', 'Flow Packets/s'
    ]
    
    def __init__(self, pcap_path: str, is_attack: bool = True):
        """
        Args:
            pcap_path: Path to PCAP file
            is_attack: Whether this traffic is attack (1) or benign (0)
        """
        self.pcap_path = pcap_path
        self.is_attack = is_attack
        self.flows = defaultdict(self._create_flow_dict)
        
    def _create_flow_dict(self):
        """Create empty flow dictionary"""
        return {
            'packets': [],
            'timestamps': [],
            'packet_sizes': [],
            'fwd_packet_sizes': [],
            'bwd_packet_sizes': [],
            'flags': defaultdict(int),
            'first_timestamp': None,
            'last_timestamp': None,
        }
    
    def extract_flows(self) -> pd.DataFrame:
        """
        Extract flows from PCAP file with computed features
        
        Returns:
            DataFrame with 11 features ready for model inference
        """
        logger.info(f"Reading PCAP: {self.pcap_path}")
        
        try:
            packets = rdpcap(self.pcap_path)
        except Exception as e:
            logger.error(f"Failed to read PCAP: {e}")
            raise
        
        logger.info(f"Processing {len(packets)} packets")
        
        # Extract flows
        for packet in packets:
            if IP not in packet:
                continue
            
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            
            # Protocol
            protocol = 'OTHER'
            if TCP in packet:
                protocol = 'TCP'
            elif UDP in packet:
                protocol = 'UDP'
            elif ICMP in packet:
                protocol = 'ICMP'
            
            # Flow key
            flow_key = (src_ip, dst_ip, protocol)
            
            # Packet timestamp
            timestamp = datetime.fromtimestamp(float(packet.time))
            
            # Extract features from this packet
            packet_size = len(packet)
            
            # Extract TCP/UDP ports if available
            sport = None
            dport = None
            flags_str = ''
            
            if TCP in packet:
                tcp_layer = packet[TCP]
                sport = tcp_layer.sport
                dport = tcp_layer.dport
                
                # TCP flags
                if tcp_layer.flags.F:
                    self.flows[flow_key]['flags']['FIN'] += 1
                if tcp_layer.flags.S:
                    self.flows[flow_key]['flags']['SYN'] += 1
                if tcp_layer.flags.R:
                    self.flows[flow_key]['flags']['RST'] += 1
                if tcp_layer.flags.P:
                    self.flows[flow_key]['flags']['PSH'] += 1
                if tcp_layer.flags.A:
                    self.flows[flow_key]['flags']['ACK'] += 1
                if tcp_layer.flags.U:
                    self.flows[flow_key]['flags']['URG'] += 1
                if tcp_layer.flags.E:
                    self.flows[flow_key]['flags']['ECE'] += 1
                if tcp_layer.flags.C:
                    self.flows[flow_key]['flags']['CWE'] += 1
                    
            elif UDP in packet:
                udp_layer = packet[UDP]
                sport = udp_layer.sport
                dport = udp_layer.dport
            
            # Store packet info
            self.flows[flow_key]['packets'].append({
                'timestamp': timestamp,
                'size': packet_size,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'sport': sport,
                'dport': dport,
            })
            
            self.flows[flow_key]['timestamps'].append(timestamp)
            self.flows[flow_key]['packet_sizes'].append(packet_size)
            
            # Directional packet sizes (simple heuristic)
            if src_ip < dst_ip:
                self.flows[flow_key]['fwd_packet_sizes'].append(packet_size)
            else:
                self.flows[flow_key]['bwd_packet_sizes'].append(packet_size)
        
        logger.info(f"Extracted {len(self.flows)} flows")
        
        # Convert flows to features
        return self._flows_to_features()
    
    def _flows_to_features(self) -> pd.DataFrame:
        """Convert extracted flows to model-compatible features"""
        
        features_list = []
        
        for (src_ip, dst_ip, protocol), flow_data in self.flows.items():
            if not flow_data['packets']:
                continue
            
            packets = flow_data['packets']
            timestamps = flow_data['timestamps']
            packet_sizes = flow_data['packet_sizes']
            
            if len(packets) < 1:
                continue
            
            # Time range
            start_time = min(timestamps)
            end_time = max(timestamps)
            flow_duration = (end_time - start_time).total_seconds()
            
            # Handle zero duration
            if flow_duration == 0:
                flow_duration = 0.001  # Avoid division by zero

            
            # Packet count
            packet_count = len(packets)
            
            # Size statistics
            avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0
            
            # Compute packets/sec
            flow_packets_per_sec = packet_count / max(flow_duration, 1)
            
            # Backward packets/sec (simple estimate)
            bwd_packets = len(flow_data['bwd_packet_sizes'])
            bwd_packets_per_sec = bwd_packets / max(flow_duration, 1)
            
            # Extract 11 features for the model
            features = {
                'Source IP': src_ip,
                'Destination IP': dst_ip,
                'Timestamp': start_time.isoformat(),
                'Label': 1 if self.is_attack else 0,  # 1 = attack, 0 = benign
                'Average Packet Size': avg_packet_size,
                'Bwd Packets/s': bwd_packets_per_sec,
                'FIN Flag Count': flow_data['flags'].get('FIN', 0),
                'SYN Flag Count': flow_data['flags'].get('SYN', 0),
                'RST Flag Count': flow_data['flags'].get('RST', 0),
                'PSH Flag Count': flow_data['flags'].get('PSH', 0),
                'ACK Flag Count': flow_data['flags'].get('ACK', 0),
                'URG Flag Count': flow_data['flags'].get('URG', 0),
                'CWE Flag Count': flow_data['flags'].get('CWE', 0),
                'ECE Flag Count': flow_data['flags'].get('ECE', 0),
                'Flow Packets/s': flow_packets_per_sec,
            }
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Fill missing values
        for col in self.FEATURE_COLS[4:]:  # Skip non-numeric columns
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        logger.info(f"Generated {len(df)} flow features")
        return df[self.FEATURE_COLS]


def pcap_to_csv(pcap_path: str, output_csv: str, is_attack: bool = True):
    """
    Convert PCAP file to CSV dataset
    
    Args:
        pcap_path: Path to input PCAP
        output_csv: Path to output CSV
        is_attack: Label (1 = attack, 0 = benign)
    
    Returns:
        Generated DataFrame
    """
    logger.info(f"Converting PCAP to CSV: {pcap_path} -> {output_csv}")
    
    extractor = FlowExtractor(pcap_path, is_attack=is_attack)
    df = extractor.extract_flows()
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved {len(df)} flows to {output_csv}")
    
    return df
