import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int):
    ddp_setup(rank=rank, world_size=world_size)

    # build dataset
    dataset = torch.rand(8)

    # build sampler
    sampler = DistributedSampler(dataset=dataset)

    for idx in sampler:
        print(f'rank{rank} get data_index: {idx}')
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'world_size: {world_size}')
    mp.spawn(main, args=(world_size,), nprocs=world_size)