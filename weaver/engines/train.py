import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial

from collections.abc import Iterator
from torch.utils.tensorboard import SummaryWriter
from weaver.datasets import build_dataset, point_collate_fn, collate_fn
from weaver.models import build_model
from weaver.utils.logger import get_root_logger
from weaver.utils.optimizer import build_optimizer
from weaver.utils.scheduler import build_scheduler
from weaver.utils.events import EventStorage, ExceptionWriter
from weaver.utils.registry import Registry