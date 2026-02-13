import time
import json
import redis, os
from scapy.all import sniff, get_if_list
from flow_extractor import FlowExtractor

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
STREAM_NAME = "ddos_stream"
WINDOW_SECONDS = 5
INTERFACE = os.getenv("CAPTURE_INTERFACE", "eth0")

def make_redis():
    return redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

r = make_redis()

extractor = FlowExtractor()
start_time = time.time()

def packet_handler(packet):
    extractor.process_packet(packet)

print("Starting real-time capture...")
print(f"Redis host: {REDIS_HOST} | Stream: {STREAM_NAME}")
print(f"Available interfaces: {get_if_list()}")
try:
    r.ping()
    print("Redis ping: ok")
except Exception as e:
    print(f"Redis ping: failed ({e})")

while True:
    try:
        sniff(prn=packet_handler, iface=INTERFACE, timeout=WINDOW_SECONDS, store=False)
    except Exception as e:
        print(f"capture error on iface {INTERFACE}: {e}")
        # fallback: try eth0 (common), then any non-lo interface.
        try:
            if INTERFACE != "eth0":
                INTERFACE = "eth0"
            else:
                for cand in get_if_list():
                    if cand != "lo":
                        INTERFACE = cand
                        break
            print(f"switching capture interface to {INTERFACE}")
        except Exception:
            pass
        time.sleep(1)
        continue

    df = extractor.build_dataframe()

    if not df.empty:
        payload = {
            "timestamp": time.time(),
            "flows": df.to_dict(orient="records")
        }

        try:
            r.xadd(STREAM_NAME, {"payload": json.dumps(payload)}, maxlen=1000, approximate=True)
        except Exception as e:
            print(f"redis publish error: {e}")
            # reconnect (covers DNS/redis restart scenarios)
            r = make_redis()
            time.sleep(1)
            continue

        print(f"Sent {len(df)} flows to Redis")

    extractor = FlowExtractor()
