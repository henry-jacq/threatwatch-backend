FROM python:3.10-slim

WORKDIR /agent

RUN apt-get update && apt-get install -y \
    tcpdump \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY agent.py .
COPY flow_extractor.py .
RUN pip install scapy redis pandas numpy

CMD ["python", "agent.py"]
