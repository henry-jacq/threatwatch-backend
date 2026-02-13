#!/bin/sh
set -e

mkdir -p /run/nginx

# Start nginx in background (daemon mode).
nginx

# Run agent in foreground so container health reflects capture health.
cd /opt/agent
exec python -u agent.py
