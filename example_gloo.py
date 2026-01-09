import torch
import ray
from ray.experimental.collective import create_collective_group


@ray.remote
class MyActor:
    @ray.method(tensor_transport="gloo")
    def random_tensor(self):
        return torch.randn(1000, 1000)

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

sender, receiver = MyActor.remote(), MyActor.remote()
group = create_collective_group([sender, receiver], backend="torch_gloo")

# The tensor will be stored by the `sender` actor instead of in Ray's object
# store.
tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
print(ray.get(result))