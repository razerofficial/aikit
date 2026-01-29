import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.v1.worker.gpu_worker import Worker as OriginWorker


class RazerGPUWorker(OriginWorker):
    def load_model(self) -> None:
        print("RazerGPUWorker load_model")
        super().load_model()

    def init_device(self):
        super().init_device()

        print(f"gpu_worker -- init_device device : cuda:{self.local_rank}")
        print("gpu_worker -- init_device init snapshot :", self.init_snapshot)
        print("gpu_worker -- init_device requested_memory :", self.requested_memory)
        self.get_gpu_info()

    def get_gpu_info(self) -> str:
        import os
        import torch
        props = torch.cuda.get_device_properties(self.device)
        uuid = "GPU-" + str(props.uuid)

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        physical_id = -1
        if cuda_visible:
            phy_devices = [int(d) for d in cuda_visible.split(",") if d.strip()]

            if (self.device.index >= 0) and (self.device.index < len(phy_devices)):
                physical_id = phy_devices[self.device.index]

            pid = os.getpid()
            print(f"CUDA_VISIBLE_DEVICES = {cuda_visible} pid = {pid}")

        print("Hello from get_gpu_info")
        print(f"physical_device : {physical_id}")
        print(f"name : {props.name}")
        print(f"UUID : {uuid}")
        return f"Hello from get_gpu_info : \n physical_device : {physical_id} \n name : {props.name} \n uuid : {uuid}"


gpu_worker.Worker = RazerGPUWorker
