import torch
import ray

# Connect to existing Ray cluster
ray.init(address='auto')

@ray.remote(num_gpus=1)
class MyActor:
    @ray.method(tensor_transport="uccl")
    def random_tensor(self):
        return torch.randn(1000, 1000).cuda()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

    def get_node_info(self):
        import socket
        import os
        return {
            'hostname': socket.gethostname(),
            'ip': socket.gethostbyname(socket.gethostname()),
            'pid': os.getpid(),
            'gpu_ids': ray.get_gpu_ids()
        }

# Get all available nodes
nodes = ray.nodes()
print(f"Available nodes: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"  Node {i}: {node['NodeManagerAddress']}")

if len(nodes) < 2:
    print("\nWarning: Only 1 node available. RDMA won't be tested.")
    print("Start a worker node with: ray start --address='<head-node-ip>:6379' --num-gpus=1")
    exit(1)

# Create actors on different nodes
print("\nCreating actors on different nodes...")
sender = MyActor.remote()
receiver = MyActor.remote()

sender_info = ray.get(sender.get_node_info.remote())
receiver_info = ray.get(receiver.get_node_info.remote())

print(f"\nSender: {sender_info}")
print(f"Receiver: {receiver_info}")

if sender_info['ip'] == receiver_info['ip']:
    print("\nWarning: Both actors on same machine - will use IPC, not RDMA")
else:
    print("\n✓ Actors on different machines - will use RDMA")

print("\nStarting UCCL tensor transfer...")
tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)

try:
    sum_result = ray.get(result, timeout=60)
    print(f"\n✓ Success! Sum result: {sum_result}")
except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
