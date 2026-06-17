# Connecting a real CSI capture device (Phase 2)

This platform classifies **WiFi Channel State Information (CSI)** — the per-subcarrier
amplitude/phase distortion a WiFi link experiences as people move through it.

> **Reality check:** stock consumer routers, laptops, and phones do **not** expose
> CSI. You cannot "plug in any router." CSI extraction requires specific hardware
> with patched firmware. This doc covers the supported devices, plus a
> zero-hardware path so anyone can exercise the full pipeline.

## How it fits together

```
capture device ──(USB serial / UDP)──► wifipose-agent ──HTTPS──► /api/v1/ingest/window
   (ESP32 / Pi)        raw CSI            (windows CSI,             (auth: X-API-Key)
                                          normalizes at edge)            │
                                                                         ▼
                                              Redis Stream ─► worker ─► classify ─► DB
                                                                         │
                                                              live WS (per device)
```

The agent does the windowing at the edge; the server normalizes whatever format
you send into the model's `[30, 104]` contract (see
[`adapter.py`](../src/wifipose/ingest/adapter.py)) and tags every result with your
`device_id` so your stream is isolated from other users'.

## 1. Register a device → get an API key

Either click **"Connect your CSI device"** in the web UI, or:

```bash
pip install "wifipose[agent]"
wifipose-agent register --server https://<your-app> --name "living-room-esp32"
# prints: device_id, api_key (shown ONCE)
```

## 2. Stream CSI

### Option A — No hardware (synthetic)
Proves the whole pipeline end-to-end with zero equipment:
```bash
wifipose-agent stream --server https://<your-app> --api-key wfp_xxx --source synthetic
```

### Option B — ESP32 (recommended real hardware, ~$5)
1. Flash an ESP32 with the **[esp-csi](https://github.com/espressif/esp-csi)** `csi_recv`
   example (Espressif's official CSI toolkit). It prints `CSI_DATA,...,[ints]` lines
   over USB serial.
2. Find the serial port (`COM3` on Windows, `/dev/ttyUSB0` on Linux).
3. Stream:
```bash
wifipose-agent stream --server https://<your-app> --api-key wfp_xxx \
  --source esp32 --port /dev/ttyUSB0 --baud 921600
```
The agent auto-parses esp-csi lines (interleaved imag/real int8 per subcarrier),
builds 30-frame windows, and posts them. `csi_format` defaults to
`complex_interleaved`.

### Option C — Replay a capture file
Replay an esp-csi log or a CSV of numbers (one packet per line):
```bash
wifipose-agent stream --server https://<your-app> --api-key wfp_xxx \
  --source file --file capture.csv --rate-hz 20
```

### Other hardware
- **Nexmon CSI** (Raspberry Pi 3B+/4, some Broadcom routers): emit CSI as UDP/CSV
  and feed it via the `file` source or a small custom source. Use
  `--format complex_pairs` or `amp_phase` to match your extractor's output.
- **Intel 5300 / Atheros CSI Tool**: export to CSV and replay via the `file` source.

## 3. Watch it live

Open the app; the live skeleton view streams predictions. To watch one device
specifically, the WebSocket accepts a filter: `/api/v1/ws?mode=live&device=<device_id>`.
Analytics are filterable too: `/api/v1/events?device_id=<device_id>`.

## Wire format (if you build your own client)

`POST /api/v1/ingest/window` with header `X-API-Key: wfp_...`:
```json
{
  "packets": [[...], [...], "... 30 packets ..."],
  "format": "complex_interleaved",
  "subject_id": "person-1",
  "frame": 0
}
```
Each packet is one CSI reading; supported `format` values:
`complex_interleaved` (ESP32 `[im,re,im,re,...]`), `complex_pairs` (`[[re,im],...]`),
`amplitude`, `amp_phase`. The server pads/trims subcarriers to 52 and the window to
30 timesteps. Continuous streaming: `WS /api/v1/ingest/stream?api_key=wfp_...`.

## ⚠️ On accuracy (read this)

The **deployed model is trained on synthetic CSI**, so feeding it real ESP32 CSI
will produce live predictions, but the **pose/action accuracy on real signals is
not validated** — there is a real domain gap between synthetic and hardware CSI
(scale, noise, antenna geometry, subcarrier count). To make real-CSI predictions
meaningful you must **retrain on real labelled CSI** via the real-data path
([`dataset.py`](../src/wifipose/ml/dataset.py), `wifipose-train`) or your own
captures. Treat the live skeleton as an **illustrative** rendering of model output,
not a calibrated body pose. The *systems* (capture → stream → classify → store →
visualize, multi-tenant, authenticated) are real; the model quality on real CSI is
future work.

## Free-tier note

Each ingested window costs a few Redis commands. Upstash's free tier is
~10k commands/day, so a fast continuous stream will exhaust it quickly. For
sustained capture use Railway's own Redis (no daily cap) or self-hosted Redis, or
lower the agent rate (`--rate-hz`) and increase `--stride`.
