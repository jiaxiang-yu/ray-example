"""
Simple UCCL RDMA test without Ray tensor transport.
Tests basic RDMA connectivity between two nodes.

Usage:
    # On receiver node:
    python test_uccl_rdma_simple.py --role receiver
    # Note the port printed in output: "Listening on X.X.X.X:PORT"

    # On sender node (after receiver is ready):
    python test_uccl_rdma_simple.py --role sender --receiver-ip <RECEIVER_IP> --receiver-port <PORT>
"""
import sys
import os

# Debug: print import info
print(f"Python: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path[:3]}...")

import argparse
import torch
from uccl.p2p import Endpoint


def run_sender(receiver_ip: str, receiver_port: int):
    print(f"[Sender] Starting...")

    # Create endpoint
    endpoint = Endpoint(0, 4)
    local_meta = endpoint.get_metadata()
    local_ip, local_port, local_gpu = Endpoint.parse_metadata(local_meta)
    print(f"[Sender] Local endpoint: {local_ip}:{local_port}, GPU {local_gpu}")

    # Create tensor
    tensor = torch.randn(1000, 1000, device="cuda")
    print(f"[Sender] Created tensor: shape={tensor.shape}, sum={tensor.sum().item():.2f}")

    # Register memory
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ok, mr_id = endpoint.reg(ptr, size)
    if not ok:
        raise RuntimeError("Failed to register memory")
    print(f"[Sender] Registered memory: mr_id={mr_id}, size={size}")

    # Connect to receiver
    print(f"[Sender] Connecting to {receiver_ip}:{receiver_port}...")
    ok, conn_id = endpoint.connect(receiver_ip, remote_gpu_idx=0, remote_port=receiver_port)
    if not ok:
        raise RuntimeError(f"Failed to connect to {receiver_ip}:{receiver_port}")
    print(f"[Sender] Connected: conn_id={conn_id}")

    # Send tensor
    print(f"[Sender] Sending tensor...")
    ok, transfer_id = endpoint.send_async(conn_id, mr_id, ptr, size)
    if not ok:
        raise RuntimeError("Failed to initiate send")

    # Poll for completion
    while True:
        ok, is_done = endpoint.poll_async(transfer_id)
        if not ok:
            raise RuntimeError("Poll failed")
        if is_done:
            break

    print(f"[Sender] Send complete!")

    # Cleanup
    endpoint.dereg(mr_id)
    print(f"[Sender] Done")


def run_receiver():
    print(f"[Receiver] Starting...")

    # Create endpoint with specific port
    import os
    os.environ["UCCL_PORT"] = "50000"  # Try to force port
    endpoint = Endpoint(0, 4)
    local_meta = endpoint.get_metadata()
    local_ip, local_port, local_gpu = Endpoint.parse_metadata(local_meta)
    print(f"[Receiver] Listening on {local_ip}:{local_port}, GPU {local_gpu}")
    print(f"[Receiver] Waiting for connection...")

    # Accept connection
    ok, remote_ip, remote_gpu, conn_id = endpoint.accept()
    if not ok:
        raise RuntimeError("Failed to accept connection")
    print(f"[Receiver] Accepted connection from {remote_ip}, GPU {remote_gpu}, conn_id={conn_id}")

    # Create receive buffer
    tensor = torch.zeros(1000, 1000, device="cuda")
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()

    # Register memory
    ok, mr_id = endpoint.reg(ptr, size)
    if not ok:
        raise RuntimeError("Failed to register memory")
    print(f"[Receiver] Registered memory: mr_id={mr_id}, size={size}")

    # Receive tensor
    print(f"[Receiver] Receiving tensor...")
    ok, transfer_id = endpoint.recv_async(conn_id, mr_id, ptr, size)
    if not ok:
        raise RuntimeError("Failed to initiate receive")

    # Poll for completion
    while True:
        ok, is_done = endpoint.poll_async(transfer_id)
        if not ok:
            raise RuntimeError("Poll failed")
        if is_done:
            break

    print(f"[Receiver] Receive complete!")
    print(f"[Receiver] Tensor sum: {tensor.sum().item():.2f}")

    # Cleanup
    endpoint.dereg(mr_id)
    print(f"[Receiver] Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--receiver-ip", help="Receiver IP (for sender)")
    parser.add_argument("--receiver-port", type=int, help="Receiver port (for sender)")
    args = parser.parse_args()

    if args.role == "sender":
        if not args.receiver_ip or not args.receiver_port:
            parser.error("--receiver-ip and --receiver-port required for sender")
        run_sender(args.receiver_ip, args.receiver_port)
    else:
        run_receiver()


if __name__ == "__main__":
    main()
