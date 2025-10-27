# ThreatWatch – DDoS Detection Platform

**Real-time DDoS detection using FTG-NET (Graph Neural Networks) with a FastAPI backend**

ThreatWatch is a production-grade system designed to detect Distributed Denial of Service (DDoS) attacks in real time using temporal flow graphs and deep learning.
It supports both unlabeled data inference and labeled data evaluation with detailed performance metrics.

---

## Overview

**Key Highlights**

* 99% accuracy using FTG-NET trained on the CICIDS2019 dataset
* Real-time inference with GPU acceleration (CUDA support)
* Dual endpoints for prediction (unlabeled) and evaluation (labeled)
* PCAP file conversion to model-compatible CSV
* Comprehensive logging and error handling
* Interactive Swagger API documentation
* Modular and scalable architecture

---

## Project Structure

```
ddos-detection-platform/
├── app/
│   ├── api/routes/          # API endpoints
│   ├── core/ml/             # ML model loading & inference
│   ├── core/traffic/        # Network flow extraction
│   ├── core/attacks/        # DDoS simulation tools
│   ├── models/              # Pydantic schemas
│   ├── utils/               # Helpers & logging
│   ├── config.py            # Configuration management
│   └── main.py              # FastAPI entry point
├── models/checkpoints_v4_metadata/
│   └── best_model_1.pt      # Trained FTG-NET model
├── attack_tools/            # SYN, UDP, HTTP flood scripts
├── scripts/pcap_convert.py  # CLI tool for PCAP conversion
├── requirements.txt
├── docker-compose.yml
├── .env
├── run.sh
└── SETUP.md
```

---

## Quick Start

### Prerequisites

* Python 3.10 or higher
* (Optional) CUDA 12.1+ for GPU acceleration
* Git

### Installation

```bash
git clone <repo_url>
cd threatwatch-backend

python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Running the Server

```bash
chmod +x run.sh
./run.sh
```

Expected output:

```
DDoS Detection Platform - Local Testing
Starting FastAPI application...
Uvicorn running on http://0.0.0.0:8000
```

**Access API:**

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* Root: [http://localhost:8000/](http://localhost:8000/)
* Health: [http://localhost:8000/api/inference/health](http://localhost:8000/api/inference/health)

---

## API Endpoints

### 1. Prediction (Unlabeled Data)

**POST** `/api/inference/predict`

Upload a CSV without a Label column.

```bash
curl -X POST -F "file=@raw_traffic.csv" \
  http://localhost:8000/api/inference/predict
```

**Response Example:**

```json
{
  "total_samples": 1352,
  "attack_count": 1324,
  "benign_count": 28,
  "average_confidence": 0.985,
  "processing_time_ms": 23451
}
```

---

### 2. Evaluation (Labeled Data)

**POST** `/api/inference/evaluate`

Upload a CSV with a Label column.

```bash
curl -X POST -F "file=@test_data.csv" \
  http://localhost:8000/api/inference/evaluate
```

**Response Example:**

```json
{
  "accuracy": 0.990,
  "precision": 0.990,
  "recall": 0.999,
  "f1_score": 0.995
}
```

---

### 3. PCAP Conversion

**POST** `/api/pcap/convert`

Convert a PCAP file to CSV format.

```bash
curl -X POST -F "file=@traffic.pcap" \
  "http://localhost:8000/api/pcap/convert?is_attack=true&output_name=ddos_traffic"
```

---

## Configuration

`.env` file example:

```env
DEBUG=True
API_VERSION=v1
MODEL_CHECKPOINT=models/checkpoints_v4_metadata/best_model_1.pt
DEVICE=auto
HOST=0.0.0.0
PORT=8000
RELOAD=True
```

---

## CSV Format Requirements

**For `/predict` (unlabeled)**
Must include:

```
Source IP, Destination IP, Timestamp, Average Packet Size, Flow Packets/s, SYN Flag Count, ACK Flag Count, etc.
```

**For `/evaluate` (labeled)**
Same as above, plus:

```
Label (values: BENIGN/0 for benign, 1 for attack)
```

---

## Testing

**Model Evaluation**

```bash
curl -X POST -F "file=@test_data.csv" \
  http://localhost:8000/api/inference/evaluate
```

**Unlabeled Prediction**

```bash
curl -X POST -F "file=@raw_traffic.csv" \
  http://localhost:8000/api/inference/predict
```

**Health Check**

```bash
curl http://localhost:8000/api/inference/health
```

---

## Docker Deployment

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Model Performance

**FTG-NET Metrics (CICIDS2019):**

* Accuracy: 99.0%
* Precision: 99.0%
* Recall: 99.9%
* F1-Score: 99.5%
* ROC-AUC: 0.99+

**Inference Speed**

* CPU: ~200 ms per graph
* GPU: ~50 ms per graph

---

## Attack Simulation Tools

**SYN Flood**

```bash
python attack_tools/syn_flood.py --target 192.168.1.1 --port 80 --duration 10
```

**UDP Flood**

```bash
python attack_tools/udp_flood.py --target 192.168.1.1 --port 53 --duration 10
```

**HTTP Flood**

```bash
python attack_tools/http_flood.py --target http://localhost:8000 --duration 10
```

---

## Logging Example

```
2025-10-27 05:16:58 - INFO - Evaluation request: test_data.csv
2025-10-27 05:17:23 - INFO - Evaluation completed: Acc=0.990, P=0.990, R=0.999, F1=0.995
```

---

## Troubleshooting

| Issue                 | Solution                                              |
| --------------------- | ----------------------------------------------------- |
| CUDA out of memory    | Set `DEVICE=cpu` in `.env`                            |
| Timestamp parse error | Ensure timestamps are ISO-formatted                   |
| Missing columns       | Check CSV against required fields                     |
| Model not found       | Verify model path in `.env`                           |
| Port in use           | Kill the process using port 8000 and rerun `./run.sh` |

---

## Core Dependencies

* FastAPI
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Pandas
* Scapy

Refer to `requirements.txt` for full list.

---

## License

This project is intended for **research and educational use only**.

---

## Citation

```
FTG-NET: Temporal Flow Graph Neural Networks for DDoS Detection
Trained on CICIDS2019 Dataset, achieving 99% accuracy
```

---

## Future Enhancements

* Real-time Kafka streaming
* WebSocket live updates
* PostgreSQL / TimescaleDB integration
* Kubernetes deployment
* React-based dashboard
* Model versioning and A/B testing

