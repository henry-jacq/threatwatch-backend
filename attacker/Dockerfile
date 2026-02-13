FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y hping3 iputils-ping curl procps && \
    rm -rf /var/lib/apt/lists/*

CMD ["sleep", "infinity"]
