#!/usr/bin/env python3
import sys
from nfstream import NFStreamer

FEATURE_ORDER = [
    "Average Packet Size", "Bwd Packets/s", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Flow Packets/s"
]

def safe_get(flow, *names, default=0):
    """Try multiple attribute names on a flow, return first non-exception value."""
    for n in names:
        try:
            v = getattr(flow, n)
            # If it is callable, attempt to call
            if callable(v):
                try:
                    v = v()
                except Exception:
                    pass
            return v if v is not None else default
        except Exception:
            continue
    return default

def flow_to_features(flow):
    # basic counters (bidirectional)
    bidir_pkts = safe_get(flow,
        "bidirectional_packets", "bidirectional_pkts", "bidirectional_packets_count", default=0)
    bidir_bytes = safe_get(flow,
        "bidirectional_bytes", "bidirectional_byte_count", "bidirectional_total_bytes", default=0)

    # duration in ms -> convert to seconds
    bidir_dur_ms = safe_get(flow,
        "bidirectional_duration_ms", "bidirectional_duration", default=0)
    # fallback: compute from first/last seen if available
    if not bidir_dur_ms:
        first = safe_get(flow, "bidirectional_first_seen_ms", "src2dst_first_seen_ms", default=0)
        last  = safe_get(flow, "bidirectional_last_seen_ms",  "src2dst_last_seen_ms", default=0)
        if first and last and last >= first:
            bidir_dur_ms = last - first
    bidir_dur_s = float(bidir_dur_ms) / 1000.0 if bidir_dur_ms else 0.0

    # backward packets (dst->src)
    dst2src_pkts = safe_get(flow, "dst2src_packets", "dst2src_pkts", "dst2src_packets_count", default=0)

    # flag counters (try several possible attribute names)
    fin = safe_get(flow, "bidirectional_fin_packets", "bidirectional_fin_flags", "bidirectional_fin", default=0)
    syn = safe_get(flow, "bidirectional_syn_packets", "bidirectional_syn_flags", "bidirectional_syn", default=0)
    rst = safe_get(flow, "bidirectional_rst_packets", "bidirectional_rst_flags", "bidirectional_rst", default=0)
    psh = safe_get(flow, "bidirectional_psh_packets", "bidirectional_psh_flags", "bidirectional_psh", default=0)
    ack = safe_get(flow, "bidirectional_ack_packets", "bidirectional_ack_flags", "bidirectional_ack", default=0)
    urg = safe_get(flow, "bidirectional_urg_packets", "bidirectional_urg_flags", "bidirectional_urg", default=0)
    cwr = safe_get(flow, "bidirectional_cwr_packets", "bidirectional_cwe_flags", "bidirectional_cwe_packets", default=0)
    ece = safe_get(flow, "bidirectional_ece_packets", "bidirectional_ece_flags", default=0)

    # computed features
    avg_pkt_size = float(bidir_bytes) / float(bidir_pkts) if (bidir_pkts and bidir_bytes) else 0.0
    bwd_pkts_per_s = float(dst2src_pkts) / bidir_dur_s if (bidir_dur_s and dst2src_pkts) else 0.0
    flow_pkts_per_s = float(bidir_pkts) / bidir_dur_s if (bidir_dur_s and bidir_pkts) else 0.0

    return [
        avg_pkt_size,
        bwd_pkts_per_s,
        int(fin),
        int(syn),
        int(rst),
        int(psh),
        int(ack),
        int(urg),
        int(cwr),
        int(ece),
        flow_pkts_per_s
    ]

def main():
    if len(sys.argv) < 2:
        print("Usage: sudo venv/bin/python nf_print_features.py <iface> [limit]")
        print("Example: sudo venv/bin/python nf_print_features.py eth0 0   (0 = infinite)")
        sys.exit(1)

    iface = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # NFStreamer instance: low-latency export settings
    streamer = NFStreamer(
        source=iface,
        active_timeout=5,       # export long-lived flows after 5s
        statistical_analysis=True
    )

    # Print CSV header
    print(",".join(f'"{h}"' for h in FEATURE_ORDER))

    count = 0
    try:
        for flow in streamer:
            row = flow_to_features(flow)
            print(",".join(str(x) for x in row))
            count += 1
            if limit and count >= limit:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
