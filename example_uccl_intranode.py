import torch
import ray

@ray.remote(num_gpus=1)
class MyActor:
    @ray.method(tensor_transport="uccl")
    def random_tensor(self):
        return torch.randn(1000, 1000).cuda()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)

    def get_gpu_info(self):
        import os
        return {
            "device_index": torch.cuda.current_device(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

sender, receiver = MyActor.remote(), MyActor.remote()

tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
print(ray.get(result))
print(ray.get(sender.get_gpu_info.remote()))
print(ray.get(receiver.get_gpu_info.remote()))