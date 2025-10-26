#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🚀 DDoS Detection Platform - Local Testing${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -r requirements.txt -q

# Check models
if [ ! -f "models/checkpoints_v4_metadata/best_model_1.pt" ]; then
    echo -e "${RED}❌ Model not found at models/checkpoints_v4_metadata/best_model_1.pt${NC}"
    exit 1
fi

# Start app
echo -e "${GREEN}✅ Starting FastAPI application...${NC}"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
