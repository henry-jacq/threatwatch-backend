# offline_test.py
from nfstream import NFStreamer
import torch, sys
from predict import try_load_model, flow_to_graphs
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python offline_test.py <model_path> <pcap_or_csv>")
    sys.exit(1)

model_path = sys.argv[1]
datafile = sys.argv[2]

# load model (choose hidden according to your model name mapping)
from predict import MODEL_CONFIGS
model_file = model_path.split("/")[-1]
hidden = MODEL_CONFIGS[model_file]["hidden"]
model, _ = try_load_model(model_path, flow_in=11, traffic_in=hidden, hidden=hidden)

streamer = NFStreamer(source=datafile, statistical_analysis=True)
for i, flow in enumerate(streamer):
    fg, tg = flow_to_graphs(flow, traffic_in=hidden)
    with torch.no_grad():
        out = model(tg, [fg]).detach().cpu().numpy()
        prob = float(out.mean())
        cls = int(prob > 0.5)
    print(i, cls, prob)
    if i > 50: break
