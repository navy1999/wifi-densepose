"""`wifipose-agent` — capture CSI and stream it to the platform.

    # one-time: register a device, get an API key
    wifipose-agent register --server https://app.example.com --name "my-esp32"

    # stream from an ESP32 running esp-csi (serial)
    wifipose-agent stream --server https://app... --api-key wfp_xxx \
        --source esp32 --port COM3 --baud 921600

    # replay a captured CSI log (esp-csi text or CSV of numbers)
    wifipose-agent stream --server ... --api-key wfp_xxx --source file --file capture.csv

    # no hardware: synthesize CSI so the whole pipeline lights up
    wifipose-agent stream --server ... --api-key wfp_xxx --source synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from collections.abc import Iterator

import httpx

from wifipose import SEQ_LEN
from wifipose.ingest.adapter import (
    FORMAT_AMP_PHASE,
    FORMAT_COMPLEX_INTERLEAVED,
    parse_esp32_line,
)


# ── sources: each yields (packet, format) tuples ──────────────────────────────
def esp32_source(port: str, baud: int) -> Iterator[tuple[list[float], str]]:
    try:
        import serial  # pyserial — install with: pip install "wifipose[agent]"
    except ImportError:  # pragma: no cover
        raise SystemExit('ESP32 source needs pyserial:  pip install "wifipose[agent]"') from None
    ser = serial.Serial(port, baud, timeout=1)
    print(f"[agent] reading esp-csi from {port} @ {baud}", file=sys.stderr)
    while True:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        packet = parse_esp32_line(raw)
        if packet:
            yield packet, FORMAT_COMPLEX_INTERLEAVED


def file_source(path: str, fmt: str) -> Iterator[tuple[list[float], str]]:
    """Replay a capture file. esp-csi CSI_DATA lines are auto-detected; other
    lines are parsed as comma/space-separated numbers with the given format."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            esp = parse_esp32_line(line)
            if esp:
                yield esp, FORMAT_COMPLEX_INTERLEAVED
                continue
            try:
                nums = [float(x) for x in line.replace(",", " ").split()]
            except ValueError:
                continue
            if nums:
                yield nums, fmt


def synthetic_source() -> Iterator[tuple[list[float], str]]:
    """Use the platform's CSI simulator; each window row is one amp_phase packet."""
    from wifipose.ml.simulator import CSISimulator

    sim = CSISimulator()
    while True:
        sample = sim.sample()
        for row in sample.csi:  # [SEQ_LEN, 104] -> per-timestep packets
            yield row.tolist(), FORMAT_AMP_PHASE


# ── streaming loop ────────────────────────────────────────────────────────────
def run_stream(args: argparse.Namespace) -> None:
    if args.source == "esp32":
        source = esp32_source(args.port, args.baud)
    elif args.source == "file":
        source = file_source(args.file, args.format)
    else:
        source = synthetic_source()

    url = args.server.rstrip("/") + "/api/v1/ingest/window"
    headers = {"X-API-Key": args.api_key}
    interval = 1.0 / max(args.rate_hz, 0.1)

    buf: deque[list[float]] = deque(maxlen=args.seq_len)
    since_emit = 0
    frame = 0
    sent = 0
    print(f"[agent] streaming source={args.source} -> {url}", file=sys.stderr)

    with httpx.Client(timeout=15) as client:
        for packet, fmt in source:
            buf.append(packet)
            since_emit += 1
            if len(buf) == args.seq_len and since_emit >= args.stride:
                since_emit = 0
                payload = {
                    "packets": list(buf),
                    "format": fmt,
                    "subject_id": args.subject,
                    "frame": frame,
                }
                try:
                    r = client.post(url, json=payload, headers=headers)
                    ok = r.status_code == 200 and r.json().get("accepted")
                    sent += 1 if ok else 0
                    if not ok:
                        print(f"[agent] rejected: {r.status_code} {r.text[:200]}", file=sys.stderr)
                    elif sent % 10 == 0:
                        print(f"[agent] sent {sent} windows", file=sys.stderr)
                except httpx.HTTPError as e:
                    print(f"[agent] send failed: {e}", file=sys.stderr)
                frame += 1
            time.sleep(interval)


def run_register(args: argparse.Namespace) -> None:
    url = args.server.rstrip("/") + "/api/v1/devices"
    body = {"name": args.name, "owner": args.owner, "csi_format": args.format}
    with httpx.Client(timeout=15) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
    print("Device registered. Save this API key — it is shown only once:\n")
    print(f"  device_id : {data['id']}")
    print(f"  api_key   : {data['api_key']}")
    print(f"  csi_format: {data['csi_format']}\n")
    print("Stream with:")
    print(
        f"  wifipose-agent stream --server {args.server} --api-key {data['api_key']} "
        f"--source synthetic"
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="wifipose-agent", description="Stream CSI to the platform.")
    sub = p.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register a device and print an API key")
    reg.add_argument("--server", required=True)
    reg.add_argument("--name", required=True)
    reg.add_argument("--owner", default="anonymous")
    reg.add_argument("--format", default=FORMAT_COMPLEX_INTERLEAVED)
    reg.set_defaults(func=run_register)

    st = sub.add_parser("stream", help="capture CSI and stream windows")
    st.add_argument("--server", required=True)
    st.add_argument("--api-key", required=True)
    st.add_argument("--source", choices=["esp32", "file", "synthetic"], default="synthetic")
    st.add_argument("--port", default="COM3", help="serial port for esp32 source")
    st.add_argument("--baud", type=int, default=921600)
    st.add_argument("--file", help="capture file for the file source")
    st.add_argument("--format", default=FORMAT_AMP_PHASE, help="format for non-esp-csi file lines")
    st.add_argument("--subject", default="person")
    st.add_argument("--rate-hz", type=float, default=20.0, help="packet read rate (file/synthetic)")
    st.add_argument("--seq-len", type=int, default=SEQ_LEN, dest="seq_len")
    st.add_argument("--stride", type=int, default=SEQ_LEN, help="packets between emitted windows")
    st.set_defaults(func=run_stream)

    args = p.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[agent] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
