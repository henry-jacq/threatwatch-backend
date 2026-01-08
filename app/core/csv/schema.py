import pandas as pd
from fastapi import HTTPException

REQUIRED_FEATURE_HEADERS = [
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Average Packet Size",
    "Bwd Packets/s",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Flow Packets/s",
]

LABEL_HEADER = "Label"


def validate_csv_headers(df: pd.DataFrame, require_label: bool):
    received_headers = set(df.columns.str.strip())
    required_headers = set(REQUIRED_FEATURE_HEADERS)

    missing_features = sorted(required_headers - received_headers)

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required CSV headers",
                "missing_headers": missing_features,
                "required_headers": REQUIRED_FEATURE_HEADERS,
                "received_headers": sorted(received_headers),
                "hint": "Ensure the CSV matches the expected schema exactly"
            }
        )

    if require_label and LABEL_HEADER not in received_headers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Label column missing",
                "required_header": LABEL_HEADER,
                "hint": "Use /predict for unlabeled data"
            }
        )
