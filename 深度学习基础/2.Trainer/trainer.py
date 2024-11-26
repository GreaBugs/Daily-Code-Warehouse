import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

from train_one_epoch import train_one_epoch
from utils import BestMetricHolder, ModelEma, save_on_master
from nets import ShenYue
from args_baoyue import get_args_parser


def collate_fn(batch):

    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def main(args):
    # 定义训练设备
    device = args.device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建网络模型
    model = ShenYue()
    # 网络模型转移到cuda上面
    model = model.to(device)
    
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:'+str(n_parameters))
    print("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # 准备数据集
    train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                                download=True)
    test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                                download=True)

    # length长度
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    print("训练数据集的长度为：{}".format(train_dataset_size))
    print("测试数据集的长度为：{}".format(test_dataset_size))

    if distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_dataloader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=collate_fn, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset_val, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, pct_start=0.2)
    elif multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)
    
    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)     

    # 添加tensorboard
    # writer = SummaryWriter("../logs_train")
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                save_on_master(weights, checkpoint_path)
            
    # for i in range(epoch):
    #     epoch_start_time = time.time()
    #     print("*"*25, end="")
    #     print("第{}轮训练开始了".format(i+1), end="")
    #     print("*"*25)
    #     train_one_epoch()
    #     # 训练开始
    #     model.train()
    #     for data in train_dataloader:
    #         imgs, targets = data
    #         imgs = imgs.to(device)
    #         targets = targets.to(device)
    #         outputs = model(imgs)
    #         loss = criterion(outputs, targets)
    #         # 优化器优化模型
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_step += 1
    #         if (total_train_step % 100) == 0:
    #             end_time = time.time()
    #             print("耗时：{}".format(end_time - start_time))
    #             print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
    #             writer.add_scalar("train_loss", loss.item(), total_train_step)

    #     # 测试步骤开始
    #     model.eval()
    #     total_test_loss = 0
    #     total_accuracy = 0
    #     # 该语句中的内容无梯度，保证不会被调优
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             imgs, targets = data
    #             imgs = imgs.to(device)
    #             targets = targets.to(device)
    #             outputs = model(imgs)
    #             loss = criterion(outputs, targets)
    #             total_test_loss += loss.item()
    #             accuracy = (outputs.argmax(1) == targets).sum()
    #             total_accuracy += accuracy
    #     print("整体测试集上的Loss为：{}".format(total_test_loss))
    #     print("整体测试集上的正确率为：{}".format(total_accuracy/test_dataset_size))
    #     writer.add_scalar("test_loss", total_test_loss, total_test_step)
    #     writer.add_scalar("test_accuracy", total_accuracy/test_dataset_size, total_test_step)
    #     total_test_step += 1

    #     torch.save(model.state_dict(), "shenyue_{}.pth".format(i))
    #     print("模型已保存")

    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)