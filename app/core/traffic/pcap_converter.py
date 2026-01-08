"""
PCAP to CICFlowMeter-style feature extraction (SAFE TEMP FILE)
"""
import logging
import tempfile
import os
import pandas as pd

from app.core.traffic.flow_extractor import FlowExtractor

logger = logging.getLogger(__name__)


def pcap_bytes_to_dataframe(pcap_bytes: bytes, is_attack: bool = True) -> pd.DataFrame:
    """
    Convert raw PCAP bytes to DataFrame safely.
    Temp file exists ONLY during extraction.
    """

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            tmp.write(pcap_bytes)
            tmp_path = tmp.name

        logger.info("Temporary PCAP created: %s", tmp_path)

        extractor = FlowExtractor(tmp_path, is_attack=is_attack)
        df = extractor.extract_flows()

        logger.info("Extracted %d flows", len(df))
        return df

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temporary PCAP deleted: %s", tmp_path)
