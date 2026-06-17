"""Reference capture agent.

Runs on the user's machine next to their CSI-capable device (ESP32 / Pi-Nexmon
/ Intel 5300) and streams CSI windows to the platform. Also supports a `file`
replay source and a `synthetic` source so anyone can try the pipeline without
hardware. Windowing happens here at the edge; the server stays stateless.
"""
