# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import os
import platform
import shutil
import time
import warnings
import numpy as np
from stat import S_IREAD, S_IRGRP, S_IROTH

import torch
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm

from .dist_utils import use_tpu
from .epoch_based_runner import EpochBasedRunner
from .hooks.optimizer import OptimizerHook, DistOptimizerHook, TPUOptimizerHook

class TPUEpochBasedRunner(EpochBasedRunner):
    def build_data_loader(self):
        super().build_data_loader()
        device = xm.xla_device()
        self.data_loader = pl.MpDeviceLoader(self.data_loader, device)

    def register_optimizer_hook(self):
        assert use_tpu(), "Must use TPU!"
        opt_hook_builder = self.optimizer_hook_params['builder']
        if opt_hook_builder == OptimizerHook or opt_hook_builder == DistOptimizerHook:
            opt_hook_builder = TPUOptimizerHook

        optimizer_hook = opt_hook_builder(
                **self.optimizer_hook_params['builder_kwargs'])
        assert isinstance(optimizer_hook, TPUOptimizerHook),\
                "Please Use TPU Optimizer Hook!"
        self.register_hook(optimizer_hook)
