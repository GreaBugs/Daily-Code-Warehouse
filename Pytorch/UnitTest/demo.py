# -*- coding: utf-8 -*-
"""DlaTldModelTaskFlowTest."""

from copy import deepcopy
from datetime import datetime

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torchpilot.utils.cfg_parser import PyConfig
from torchpilot.utils.file_manager import FileManager
from torchpilot.utils.registries import DATALOADER
from torchpilot.utils.registries import EVALUATOR
from torchpilot.utils.registries import TASK_FLOW

def test_overwrite(
    dla_tld_owerwrite_ckpt_path,
    dla_tld_owerwrite_meta_file,
):
    """Test overwrite."""
    dla_tld_config_path = "experiments/task_dla/exp_dla/cfg_dla_tld_raw.py"

    cfg = PyConfig.fromfile(dla_tld_config_path)
    assert cfg.ckpt["pretrain_model"] == dla_tld_owerwrite_ckpt_path
    assert cfg.dataloader["dataset_cfg"]["meta_file"] == dla_tld_owerwrite_meta_file

def test_task_dla_onemodel_dla_tld_raw_dataloader_taskflow_train_one_step_tld(
    bev_onemodel_dla_tld_raw_config_path,
):
    """Test dataloader taskflow run one step."""
    cfg = PyConfig.fromfile(bev_onemodel_dla_tld_raw_config_path)
    # Init fake config.
    torch.cuda.empty_cache()
    FileManager.init(
        cfg_path=bev_onemodel_dla_tld_raw_config_path,
        run_name=f'{datetime.now().strftime("%H_%M_%S")}_dla_tld_raw',
    )

    dataloader_cfg = deepcopy(cfg.dataloader)
    dataloader_cfg["num_workers"] = 1
    dataloader_cfg["dataset_cfg"]["mode"] = "train"
    dataloader_cfg["dataset_cfg"]["meta_file"]["dataset"]["train"] = [
        "/share-global/baoyue.shen/Master_Test/tld-raw-label-v2-0112.pkl"
    ]
    task_flow_cfg = deepcopy(cfg.task_flow)
    dataloader_instance = DATALOADER.build(dataloader_cfg)
    task_flow_instance = TASK_FLOW.build(task_flow_cfg)

    task_flow_instance.cuda()
    task_flow_instance = DistributedDataParallel(task_flow_instance, device_ids=[0])

    dataloader_instance.set_state(epoch=0)
    batch_data = dataloader_instance.get_batch_data()
    model_outputs = task_flow_instance(batch_data)
    del task_flow_instance
    del dataloader_instance
    del batch_data
    del model_outputs

def test_task_dla_cfg_fp32_dataloader_taskflow_evaluator_infer_one_step_dla_tld(
    bev_onemodel_dla_tld_raw_config_path,
):
    """Test dataloader taskflow run one step."""
    cfg = PyConfig.fromfile(bev_onemodel_dla_tld_raw_config_path)
    # Init fake config.
    torch.cuda.empty_cache()
    FileManager.init(
        cfg_path=bev_onemodel_dla_tld_raw_config_path,
        run_name=f'{datetime.now().strftime("%H_%M_%S")}_dla_tld_raw',
    )

    dataloader_cfg = deepcopy(cfg.infer_dataloader)
    dataloader_cfg["num_workers"] = 1
    dataloader_cfg["dataset_cfg"]["mode"] = "val"
    dataloader_cfg["dataset_cfg"]["meta_file"]["dataset"]["val"] = [
        "/share-global/baoyue.shen/RAW_baoyue_tools/Datasets/VAL-FOR-RAW/rgb_raw_with_number_rgbval.pkl"  # noqa: E501 pylint:disable=line-too-long
    ]

    task_flow_cfg = deepcopy(cfg.infer_task_flow)
    evaluator_cfg = deepcopy(cfg.evaluator)
    dataloader_instance = DATALOADER.build(dataloader_cfg)
    task_flow_instance = TASK_FLOW.build(task_flow_cfg)
    evaluator_instance = EVALUATOR.build(evaluator_cfg)

    task_flow_instance.cuda()
    task_flow_instance = DistributedDataParallel(task_flow_instance, device_ids=[0])
    task_flow_instance.eval()
    dataloader_instance.set_state(epoch=0)
    batch_data = dataloader_instance.get_batch_data()
    with torch.no_grad():
        model_outputs = task_flow_instance(batch_data)

    eval_results = evaluator_instance.eval(model_outputs, batch_data)
    evaluator_instance.dist_all_reduce(eval_results)
    del task_flow_instance
    del dataloader_instance
    del batch_data
    del model_outputs
    del evaluator_instance

if __name__ == "__main__":
    pytest.main([__file__])
