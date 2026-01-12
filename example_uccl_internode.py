"""
Simple internode UCCL test script.

For internode (RDMA) testing:
    # On node 1 (head):
    ray start --head --node-ip-address=<HEAD_IP> --num-gpus=1

    # On node 2:
    ray start --address=<HEAD_IP>:6379 --num-gpus=1

    # Run this script on head node:
    python example_uccl_internode.py

For single-node (IPC) testing:
    python example_uccl_internode.py
"""
import torch
import ray
import time
import sys


@ray.remote(num_gpus=1)
class UCCLActor:
    def __init__(self, name: str):
        self.name = name
        self.node_id = ray.get_runtime_context().get_node_id()
        self.node_ip = ray.util.get_node_ip_address()
        print(f"[{name}] Initialized on node {self.node_id[:8]}... IP={self.node_ip}")

    def get_info(self):
        return self.node_id, self.node_ip

    @ray.method(tensor_transport="uccl")
    def create_tensor(self, size: int = 1000):
        tensor = torch.randn(size, size, device="cuda")
        print(f"[{self.name}] Created tensor shape={tensor.shape} device={tensor.device}")
        return tensor

    def receive_and_sum(self, tensor: torch.Tensor):
        print(f"[{self.name}] Received tensor shape={tensor.shape} device={tensor.device}")
        result = float(torch.sum(tensor).cpu())
        print(f"[{self.name}] Sum = {result:.2f}")
        return result

    def produce(self, tensors):
        refs = []
        for t in tensors:
            refs.append(ray.put(t, _tensor_transport="uccl"))
        return refs

    def consume_with_uccl(self, refs):
        # ray.get will also use UCCL to retrieve the
        # result.
        tensors = [ray.get(ref) for ref in refs]
        sum = 0
        for t in tensors:
            assert t.device.type == "cuda"
            sum += t.sum().item()
        return sum


def main():
    # Must connect to an existing Ray cluster for internode testing
    import os
    address = os.environ.get("RAY_ADDRESS", "auto")
    print(f"Connecting to Ray cluster at: {address}", flush=True)
    ray.init(address=address, ignore_reinit_error=True, logging_level="DEBUG")
    print("Ray init completed!", flush=True)

    nodes = ray.nodes()
    print(f"Got {len(nodes)} nodes", flush=True)
    alive_nodes = [n for n in nodes if n["Alive"]]
    print(f"\n=== Cluster Info ===")
    print(f"Nodes: {len(alive_nodes)}")
    for n in alive_nodes:
        print(f"  - {n['NodeName']} (GPUs: {n['Resources'].get('GPU', 0)})")

    if len(alive_nodes) < 2:
        print("\nWARNING: Single node - will test IPC path, not internode RDMA")
        print("To test internode: start ray on 2 nodes with --num-gpus=1 each")

    # Create actors
    print("\n=== Creating Actors ===")
    sender = UCCLActor.remote("Sender")
    receiver = UCCLActor.remote("Receiver")

    # Get actor info
    sender_id, sender_ip = ray.get(sender.get_info.remote())
    receiver_id, receiver_ip = ray.get(receiver.get_info.remote())

    print(f"\nSender:   node={sender_id[:8]}... IP={sender_ip}")
    print(f"Receiver: node={receiver_id[:8]}... IP={receiver_ip}")

    is_internode = (sender_id != receiver_id)
    print(f"\nTransport: {'RDMA (internode)' if is_internode else 'IPC (same node)'}")

    # Test transfer
    print("\n=== Testing UCCL Transfer ===", flush=True)
    start = time.time()
    print("Creating tensor on sender...", flush=True)
    tensor_ref = sender.create_tensor.remote(1000)
    print(f"Tensor ref created: {tensor_ref}", flush=True)
    print("Calling receive_and_sum on receiver...", flush=True)
    ref1 = sender.receive_and_sum.remote(tensor_ref)
    ref2 = receiver.receive_and_sum.remote(tensor_ref)
    print("Getting result...", flush=True)

    result = ray.get(ref2)
    elapsed = time.time() - start

    print(f"\n=== Result ===")
    print(f"Transfer time: {elapsed:.3f}s")
    print(f"Tensor sum: {result:.2f}")
    print("SUCCESS!")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
