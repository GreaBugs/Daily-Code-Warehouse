import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default="/home/yaohuayang/baoyue.shen/CODE/DINO/config/DINO/DINO_4scale.py", type=str)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/home/yaohuayang/baoyue.shen/CODE/DINO/COCODIR')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='/home/yaohuayang/baoyue.shen/CODE/DINO/logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=True, help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', default='', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_false')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    
    parser.add_argument('--learning_rate', default= 0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--lr_drop_list', default=[33, 45])
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--use_ema', default=True, type=bool)
    parser.add_argument('--ema_decay', default=0.9997, type=float)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser