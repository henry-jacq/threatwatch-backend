import io
import pandas as pd
from fastapi import HTTPException

def read_csv_safe(file_bytes: bytes) -> pd.DataFrame:
    """
    Robust CSV reader with encoding + delimiter handling.
    """
    text = None

    # Try decoding
    for enc in ("utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file encoding",
                "hint": "Upload a real CSV file (UTF-8 / Excel CSV). Binary files are not supported."
            }
        )

    # Parse CSV safely
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=None,              # auto-detect delimiter (, ; | \t)
            engine="python",       # REQUIRED for sep=None
            on_bad_lines="error"   # fail fast on malformed rows
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid CSV format",
                "reason": str(e),
                "hint": (
                    "Ensure the file is a valid CSV. "
                    "If using Excel: Save As → CSV (UTF-8). "
                    "Do NOT upload .xlsx, .pcap, or binary files."
                )
            }
        )

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CSV appears empty or malformed",
                "columns_detected": list(df.columns)
            }
        )

    return df

