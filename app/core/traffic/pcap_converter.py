"""
PCAP to CICFlowMeter-compatible Dataset Converter
Converts raw PCAP files to the 11-feature dataset format
"""
import logging
from app.core.traffic.flow_extractor import FlowExtractor

logger = logging.getLogger(__name__)


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
