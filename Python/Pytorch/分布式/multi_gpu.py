import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os


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


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


def main(rank: int, world_size: int, total_epochs: int, save_every: int, batch_size: int):
    # setup ddp
    ddp_setup(rank=rank, world_size=world_size)

    # load dataset
    train_dataset = MyTrainDataset(2048)

    # build dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=False,
                                  sampler=DistributedSampler(train_dataset)
                                  )

    # build model
    model = torch.nn.Linear(20, 1).to(rank)
    model = DDP(model, device_ids=[rank])

    # build optim
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(total_epochs):
        bs = len(next(iter(train_dataloader))[0])
        train_dataloader.sampler.set_epoch(epoch)
        print(f"[GPU{rank}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(train_dataloader)}")

        for source, targets in train_dataloader:
            source = source.to(rank)
            targets = targets.to(rank)

            optimizer.zero_grad()
            output = model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

        if rank == 0 and epoch % save_every == 0:
            ckpt = model.module.state_dict()
            PATH = "checkpoint.pth.tar"
            torch.save(ckpt, f=PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=5, type=int, help="Total epochs to train the model")
    parser.add_argument('--save_every', default=2, type=int, help="How often to save a snapshot")
    parser.add_argument('--batch_size', default=32, type=int, help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f'world_size: {world_size}')
    mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.batch_size), nprocs=world_size)
