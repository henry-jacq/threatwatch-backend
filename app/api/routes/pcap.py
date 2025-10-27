"""
PCAP file upload and conversion endpoints
Extracts features only - reuses /api/inference/predict for analysis
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
import logging
import os
from pathlib import Path

from app.core.traffic.pcap_converter import pcap_to_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pcap", tags=["pcap"])


@router.post("/convert")
async def convert_pcap(
    file: UploadFile = File(...),
    output_name: str = Query("converted", description="Output CSV filename (without .csv)")
):
    """
    Convert PCAP to model-compatible CSV dataset
    
    Extracts 11 features from raw network traffic.
    Use output CSV with /api/inference/predict for analysis.
    
    Args:
        file: PCAP file to convert
        output_name: Output CSV filename
    
    Returns:
        Path to extracted CSV file ready for inference
    """
    try:
        logger.info(f"Processing PCAP: {file.filename}")
        
        # Save uploaded file temporarily
        temp_pcap = f"/tmp/{file.filename}"
        contents = await file.read()
        
        with open(temp_pcap, "wb") as f:
            f.write(contents)
        
        logger.info(f"Saved temp file: {temp_pcap}")
        
        # Convert PCAP to CSV (always label as unknown/attack by default)
        output_csv = f"data/processed/{output_name}.csv"
        os.makedirs("data/processed", exist_ok=True)
        
        df = pcap_to_csv(temp_pcap, output_csv, is_attack=True)
        
        # Cleanup temp file
        os.remove(temp_pcap)
        
        logger.info(f"✅ Extraction complete: {len(df)} flows")
        
        return {
            "status": "✅ success",
            "input_file": file.filename,
            "output_file": output_csv,
            "flows_extracted": len(df),
            "columns": list(df.columns),
            "next_step": f"Upload {output_csv} to /api/inference/predict for real-time analysis"
        }
        
    except Exception as e:
        logger.error(f"PCAP conversion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PCAP conversion failed: {str(e)}")


@router.get("/stats")
async def pcap_stats(csv_file: str = Query(..., description="Path to converted CSV")):
    """Get statistics about converted dataset"""
    try:
        import pandas as pd
        
        if not os.path.exists(csv_file):
            raise HTTPException(status_code=404, detail=f"File not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        return {
            "file": csv_file,
            "total_flows": len(df),
            "average_packet_size": float(df['Average Packet Size'].mean()),
            "total_columns": len(df.columns),
            "columns": list(df.columns)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
