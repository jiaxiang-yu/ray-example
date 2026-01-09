import torch
import ray

@ray.remote(num_gpus=2)
class MyActor:
    @ray.method(tensor_transport="uccl")
    def random_tensor(self):
        return torch.randn(1000, 1000).cuda()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

# No collective group is needed. The two actors just need to have NIXL
# installed.
sender, receiver = MyActor.remote(), MyActor.remote()

# The tensor will be stored by the `sender` actor instead of in Ray's object
# store.
tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
print(ray.get(result))