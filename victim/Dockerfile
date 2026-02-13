FROM python:3.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      nginx \
      tcpdump \
      libpcap0.8 \
    && rm -rf /var/lib/apt/lists/*

# Agent deps
RUN pip install --no-cache-dir scapy redis pandas numpy

COPY agent/agent.py /opt/agent/agent.py
COPY agent/flow_extractor.py /opt/agent/flow_extractor.py

WORKDIR /opt/agent

COPY victim/start.sh /start.sh
RUN chmod +x /start.sh

ENV REDIS_HOST=redis
ENV CAPTURE_INTERFACE=eth0

EXPOSE 80

CMD ["/start.sh"]
