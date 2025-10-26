"""
Real-time packet capture and flow extraction using Scapy
"""
import logging
from scapy.all import sniff, IP, TCP, UDP, ICMP
from typing import Callable, Dict
import threading
import time

logger = logging.getLogger(__name__)


class PacketCaptureEngine:
    """Capture packets and extract flow features"""
    
    def __init__(self, interface: str = None, packet_count: int = 1000):
        self.interface = interface
        self.packet_count = packet_count
        self.packets = []
        self.flows = {}
        self.running = False
        
    def packet_callback(self, packet):
        """Process captured packets"""
        try:
            if IP in packet:
                ip_layer = packet[IP]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                protocol = ip_layer.proto
                
                # Extract transport layer info
                if TCP in packet:
                    tcp_layer = packet[TCP]
                    flow_key = (src_ip, dst_ip, tcp_layer.sport, tcp_layer.dport, 'TCP')
                elif UDP in packet:
                    udp_layer = packet[UDP]
                    flow_key = (src_ip, dst_ip, udp_layer.sport, udp_layer.dport, 'UDP')
                else:
                    flow_key = (src_ip, dst_ip, protocol)
                
                # Store packet
                self.packets.append({
                    'timestamp': time.time(),
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'protocol': protocol,
                    'size': len(packet),
                    'flow_key': flow_key
                })
                
                # Update flow
                if flow_key not in self.flows:
                    self.flows[flow_key] = {'packets': [], 'bytes': 0}
                self.flows[flow_key]['packets'].append(self.packets[-1])
                self.flows[flow_key]['bytes'] += len(packet)
                
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def start_capture(self, timeout: int = 10):
        """Start packet capture in background thread"""
        logger.info(f"🔍 Starting packet capture on {self.interface or 'default interface'}")
        self.running = True
        
        def capture_thread():
            sniff(
                prn=self.packet_callback,
                iface=self.interface,
                store=False,
                timeout=timeout,
                filter="ip"
            )
            self.running = False
            logger.info("✅ Packet capture completed")
        
        t = threading.Thread(target=capture_thread, daemon=True)
        t.start()
    
    def get_flows(self) -> Dict:
        """Get extracted flows"""
        return self.flows
    
    def clear(self):
        """Clear captured data"""
        self.packets = []
        self.flows = {}
