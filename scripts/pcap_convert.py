#!/usr/bin/env python3
"""
Standalone PCAP to CSV converter
Usage: python scripts/pcap_convert.py input.pcap output.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.traffic.pcap_converter import pcap_to_csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from PCAP file'
    )
    parser.add_argument('input_pcap', help='Path to input PCAP file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("PCAP FEATURE EXTRACTION")
    logger.info("="*70)
    logger.info(f"Input:  {args.input_pcap}")
    logger.info(f"Output: {args.output_csv}")
    
    try:
        df = pcap_to_csv(args.input_pcap, args.output_csv, is_attack=True)
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ Extraction Successful!")
        logger.info("="*70)
        logger.info(f"Flows extracted: {len(df)}")
        logger.info(f"Features: {len(df.columns)}")
        logger.info(f"\nColumns: {list(df.columns)}")
        logger.info(f"\nFirst few flows:")
        logger.info(df.head(3).to_string())
        logger.info(f"\n📊 Next: Upload {args.output_csv} to /api/inference/predict")
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
