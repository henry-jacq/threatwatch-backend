"""
PCAP upload & feature extraction (stateless, in-memory)
"""
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import logging

from app.core.traffic.pcap_converter import pcap_bytes_to_dataframe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pcap", tags=["pcap"])


# PCAP to CSV (DOWNLOAD)

@router.post("/convert")
async def convert_pcap(
    file: UploadFile = File(...),
    output_name: str = Query(
        "converted",
        description="Output filename (without .csv)"
    )
):
    """
    Convert PCAP to CSV and return as downloadable file
    """

    try:
        logger.info("Converting PCAP: %s", file.filename)

        df = pcap_bytes_to_dataframe(await file.read())

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No flows extracted from PCAP"
            )

        # Convert DataFrame to CSV (in memory)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        filename = f"{output_name}.csv"

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }

        logger.info("Returning CSV (%d flows) as %s", len(df), filename)

        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers=headers
        )

    except Exception as e:
        logger.error("PCAP conversion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# PCAP to STATS (NO CSV)

@router.post("/stats")
async def pcap_stats(file: UploadFile = File(...)):
    """
    Return statistics from PCAP without exporting CSV.
    """
    logger.info("PCAP stats request: %s", file.filename)

    df = pcap_bytes_to_dataframe(await file.read(), is_attack=True)

    return JSONResponse({
        "file": file.filename,
        "total_flows": len(df),
        "average_packet_size": float(df["Average Packet Size"].mean()),
        "columns": list(df.columns)
    })
