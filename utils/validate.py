# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from typing import Tuple, Iterable, Optional, Callable
from .metrics import ClassificationMeter


@torch.no_grad()
def validate(
    model: Callable[[torch.Tensor], torch.Tensor],
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    desc: Optional[str] = None,
    return_preds: bool = False  # 新增参数
):
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device
    else:
        device = model.device

    meter = ClassificationMeter(num_classes)

    # 如果需要返回标签和预测
    all_targets = []
    all_logits = []

    for X, y in tqdm(data_loader, desc=desc):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward
        logits: torch.Tensor = model(X)
        meter.record(y, logits)

        if return_preds:
            all_targets.append(y.detach().cpu())
            all_logits.append(logits.detach().cpu())

    if return_preds:
        all_targets = torch.cat(all_targets)
        all_logits = torch.cat(all_logits)
        meter.y_true = all_targets
        meter.y_pred = all_logits

    return meter