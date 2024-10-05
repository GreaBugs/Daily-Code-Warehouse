import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


def main(gpu_id, total_epochs, save_every, batch_size):
    # build dataset
    train_dataset = MyTrainDataset(2048)

    # build dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4)

    # build model
    model = torch.nn.Linear(20, 1).to(gpu_id)

    # build optim
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train start
    for epoch in range(total_epochs):
        bs = len(next(iter(train_dataloader))[0])
        print(f"[GPU{gpu_id}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(train_dataloader)}")

        for source, targets in train_dataloader:
            source = source.to(gpu_id)
            targets = targets.to(gpu_id)

            optimizer.zero_grad()
            output = model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

        if epoch % save_every == 0:
            ckpt = model.state_dict()
            PATH = "checkpoint.pth.tar"
            torch.save(ckpt, f=PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=5, type=int, help="Total epochs to train the model")
    parser.add_argument('--save_every', default=2, type=int, help="How often to save a snapshot")
    parser.add_argument('--batch_size', default=32, type=int, help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    gpu_id = 0
    main(gpu_id=gpu_id,
         total_epochs=args.total_epochs,
         save_every=args.save_every,
         batch_size=args.batch_size)