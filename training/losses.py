from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_CHOICES = ("msre", "mlse", "msle", "log_mse", "huber")
LOSS_EPS = 1e-12


def _masked_mean(loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weighted = loss * valid
    return weighted.sum() / valid.sum().clamp(min=1.0)


def compute_msre(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = LOSS_EPS,
) -> torch.Tensor:
    rel_err = (pred - target) / (target + eps)
    return _masked_mean(rel_err.square(), valid)


def compute_mlse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = LOSS_EPS,
) -> torch.Tensor:
    pred_log = torch.log(pred + eps)
    target_log = torch.log(target + eps)
    return _masked_mean((pred_log - target_log).square(), valid)


def compute_log_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = LOSS_EPS,
) -> torch.Tensor:
    pred_log = torch.log(torch.clamp(pred, min=eps))
    target_log = torch.log(torch.clamp(target, min=eps))
    return _masked_mean((pred_log - target_log).square(), valid)


def compute_log_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = LOSS_EPS,
    delta: float = 1.0,
) -> torch.Tensor:
    pred_log = torch.log(torch.clamp(pred, min=eps))
    target_log = torch.log(torch.clamp(target, min=eps))
    return _masked_mean(
        F.huber_loss(pred_log, target_log, reduction="none", delta=delta),
        valid,
    )


def compute_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "msre":
        return compute_msre(pred, target, valid)
    if loss_name in {"mlse", "msle"}:
        return compute_mlse(pred, target, valid)
    if loss_name == "log_mse":
        return compute_log_mse(pred, target, valid)
    if loss_name == "huber":
        return compute_log_huber(pred, target, valid)
    raise ValueError(f"Unsupported loss: {loss_name}")
