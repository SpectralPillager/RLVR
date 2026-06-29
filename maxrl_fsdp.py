#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-turn MaxRL training for RWKV.

Code provenance in this file:
- Directly reused modules/helpers:
  - FSDP/backend utilities from `sft_tool_use_fsdp.py`
  - Rollout engine and judge path from `eval_math192_grpo.py`
- Hand-copied logic:
  - `compute_maxrl_outcome_advantage` from
    `maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py`
  - `_ppo_clipped_objective` and `_sync_infer_weights` logic from
    `grpo_math_local.py`
- Original glue in this file:
  - CLI/config wiring
  - single-turn data preprocessing
  - rollout/judge-to-trajectory assembly
  - prompt-group training loop that keeps the SFT FSDP backend structure
"""

import argparse
import contextlib
import copy
import functools
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMP_DIR = os.path.join(REPO_DIR, ".local_tmp")
LOCAL_TORCH_EXTENSIONS_DIR = os.path.join(REPO_DIR, ".torch_extensions")
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_TORCH_EXTENSIONS_DIR, exist_ok=True)
os.environ["TMPDIR"] = LOCAL_TMP_DIR
os.environ["TEMP"] = LOCAL_TMP_DIR
os.environ["TMP"] = LOCAL_TMP_DIR
os.environ["TORCH_EXTENSIONS_DIR"] = LOCAL_TORCH_EXTENSIONS_DIR
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import eval_math192_grpo as eval_base
import sft_tool_use_fsdp as sft_backend
from grpo_math_local import (
    FSDP,
    MemoryEfficientAdamW,
    MixedPrecision,
    ShardingStrategy,
    _dist_is_initialized,
    _ensure_cuda_toolkit_env,
    _get_tensor_by_dotted_name,
    _init_distributed_from_env,
    _parse_cuda_index,
    _resolve_torch_cuda_arch_list,
    enable_full_finetune,
    load_data,
    load_train_model_rwkv7_cuda,
    normalize_model_arg,
    now_str,
    transformer_auto_wrap_policy,
)


MUON_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _stochastic_roundf_to_bf16(fp32_tensor: torch.Tensor) -> torch.Tensor:
    if fp32_tensor.dtype != torch.float32:
        fp32_tensor = fp32_tensor.float()
    bits = fp32_tensor.view(torch.int32)
    rnd = torch.randint(
        0,
        1 << 16,
        bits.shape,
        device=bits.device,
        dtype=torch.int32,
    )
    rounded = bits + rnd
    return (rounded & 0xFFFF0000).view(torch.float32).to(torch.bfloat16)


def _muon_zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError("Muon input must be a 2D matrix")
    if int(ns_steps) >= 100:
        raise ValueError("Muon ns_steps must be less than 100")
    a, b, c = ns_coefficients
    x = grad.bfloat16()
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / x.norm().clamp(min=float(eps))
    for _ in range(int(ns_steps)):
        xx_t = x @ x.T
        update = torch.addmm(xx_t, xx_t, xx_t, beta=float(b), alpha=float(c))
        x = torch.addmm(x, update, x, beta=float(a))
    if transposed:
        x = x.T
    return x


def _muon_adjust_lr(lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size) -> float:
    a, b = int(param_shape[0]), int(param_shape[1])
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, float(a) / max(1.0, float(b))))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(float(max(a, b)))
    else:
        ratio = 1.0
    return float(lr) * float(ratio)


class Muon2DWithAdamWFallback:
    """Muon for 2D hidden weights plus MemoryEfficientAdamW fallback."""

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        lr: float,
        adam_lr: Optional[float],
        betas: Tuple[float, float],
        adam_eps: float,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_eps: float = 1e-7,
        muon_adjust_lr_fn: Optional[str] = None,
        muon_variant: str = "torch",
        muon_weight_decay: float = 0.0,
        adam_weight_decay: float = 0.0,
        pin_memory: bool = True,
    ):
        self.param_groups = []
        self.state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        self.muon_params: List[torch.nn.Parameter] = []
        self.muon_names: List[str] = []
        self.adam_params: List[torch.nn.Parameter] = []
        self.adam_names: List[str] = []
        self.lr = float(lr)
        self.adam_lr = float(adam_lr) if adam_lr is not None and float(adam_lr) > 0.0 else self.lr
        self.muon_momentum = float(muon_momentum)
        self.muon_nesterov = bool(muon_nesterov)
        self.muon_ns_steps = int(muon_ns_steps)
        self.muon_eps = float(muon_eps)
        self.muon_adjust_lr_fn = muon_adjust_lr_fn
        self.muon_variant = str(muon_variant).lower()
        self.muon_weight_decay = float(muon_weight_decay)
        self.pin_memory = bool(pin_memory)
        if self.muon_adjust_lr_fn not in (None, "original", "match_rms_adamw", "none"):
            raise ValueError(f"Unsupported muon_adjust_lr_fn: {self.muon_adjust_lr_fn}")
        if self.muon_variant not in ("torch", "moonlight"):
            raise ValueError(f"Unsupported muon_variant: {self.muon_variant}")

        for name, p in named_params:
            if not p.requires_grad:
                continue
            lname = str(name)
            is_embed_or_head = lname in ("emb.weight", "head.weight") or lname.endswith(".emb.weight") or lname.endswith(".head.weight")
            if p.ndim == 2 and not is_embed_or_head:
                self.muon_params.append(p)
                self.muon_names.append(lname)
            else:
                self.adam_params.append(p)
                self.adam_names.append(lname)

        self.adam = None
        if self.adam_params:
            self.adam = MemoryEfficientAdamW(
                self.adam_params,
                lr=self.adam_lr,
                betas=betas,
                eps=float(adam_eps),
                weight_decay=float(adam_weight_decay),
                enabled=True,
            )
        if self.muon_params:
            self.param_groups.append({
                "params": self.muon_params,
                "lr": self.lr,
                "weight_decay": self.muon_weight_decay,
                "momentum": self.muon_momentum,
                "nesterov": self.muon_nesterov,
                "ns_coefficients": MUON_DEFAULT_NS_COEFFICIENTS,
                "eps": self.muon_eps,
                "ns_steps": self.muon_ns_steps,
                "adjust_lr_fn": self.muon_adjust_lr_fn,
                "variant": self.muon_variant,
            })
        if self.adam is not None:
            self.param_groups.extend(self.adam.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)

    def _get_muon_master(self, p: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if p.dtype == torch.float32:
            return None
        state = self.state[p]
        if "fp32_param" not in state:
            fp32_p = p.data.float().to(device="cpu", copy=True)
            if self.pin_memory:
                try:
                    fp32_p = fp32_p.pin_memory()
                except Exception:
                    pass
            state["fp32_param"] = fp32_p
        return state["fp32_param"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for p in self.muon_params:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if p.grad.ndim != 2:
                raise ValueError("Muon gradient must be 2D")
            state = self.state.setdefault(p, {})
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            grad = p.grad
            if self.muon_variant == "moonlight":
                buf.mul_(self.muon_momentum).add_(grad)
                update = grad.add(buf, alpha=self.muon_momentum) if self.muon_nesterov else buf
            else:
                buf.lerp_(grad, 1.0 - self.muon_momentum)
                update = grad.lerp(buf, self.muon_momentum) if self.muon_nesterov else buf
            update = _muon_zeropower_via_newtonschulz(
                update,
                ns_coefficients=MUON_DEFAULT_NS_COEFFICIENTS,
                ns_steps=self.muon_ns_steps,
                eps=self.muon_eps,
            ).to(device=p.device, dtype=torch.float32)
            adjust_fn = None if self.muon_adjust_lr_fn == "none" else self.muon_adjust_lr_fn
            adjusted_lr = _muon_adjust_lr(self.lr, adjust_fn, p.shape)
            fp32_p_cpu = self._get_muon_master(p)
            if fp32_p_cpu is not None:
                fp32_p = fp32_p_cpu.to(p.device, non_blocking=True)
                if self.muon_weight_decay != 0:
                    fp32_p.mul_(1.0 - self.lr * self.muon_weight_decay)
                fp32_p.add_(update, alpha=-float(adjusted_lr))
                p.data.copy_(_stochastic_roundf_to_bf16(fp32_p))
                fp32_p_cpu.copy_(fp32_p, non_blocking=True)
            else:
                if self.muon_weight_decay != 0:
                    p.data.mul_(1.0 - self.lr * self.muon_weight_decay)
                p.data.add_(update.to(dtype=p.dtype), alpha=-float(adjusted_lr))

        if self.adam is not None:
            self.adam.step()
        return loss


def _extract_gsm8k_final_answer(solution: str) -> str:
    s = str(solution or "").strip()
    if not s:
        return ""
    m = re.search(r"####\s*(.+?)\s*$", s, flags=re.DOTALL)
    final = m.group(1).strip() if m else s
    final = final.splitlines()[0].strip()
    return final.strip("$").strip()


def _last_boxed(text: str) -> Optional[str]:
    s = str(text or "")
    key = r"\boxed{"
    start = s.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    out = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _strip_math_delims(s: str) -> str:
    s = str(s or "").strip()
    changed = True
    while changed:
        old = s
        s = s.strip()
        if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) >= 2:
            s = s[1:-1].strip()
        if s.startswith(r"\(") and s.endswith(r"\)"):
            s = s[2:-2].strip()
        if s.startswith(r"\[") and s.endswith(r"\]"):
            s = s[2:-2].strip()
        changed = s != old
    return s


def _extract_answer_tail_expression(tail: str) -> Optional[str]:
    if tail is None:
        return None
    s = str(tail).lstrip()
    if not s:
        return None
    if s[0] in [":", "：", "="]:
        s = s[1:].lstrip()
    elif s[0] in [".", "。", ",", "，", ";", "；", "!", "！", "?", "？"]:
        return None
    s = s.splitlines()[0].split("</think>")[0].strip()
    lowered = s.lower()
    for pat in [
        r"^(is|are|be|equals?)\b",
        r"^(should\s+be|would\s+be|could\s+be)\b",
        r"^(is\s+equal\s+to|equal\s+to)\b",
        r"^(box|boxed)\b",
    ]:
        m = re.match(pat, lowered)
        if m:
            s = s[m.end() :].lstrip(" :：=,-")
            lowered = s.lower()
            break
    boxed = _last_boxed(s)
    if boxed:
        return boxed
    s = re.split(r"\.\s+(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]
    s = re.split(r"[;,，；:：]\s*(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]
    s = re.sub(r"[\.\。,\，;；:：!！?？]+$", "", s).strip()
    return s or None


def _extract_last_valid_answer_marker(text: str) -> Optional[str]:
    candidates = []
    for m in re.finditer(r"(?i)\banswer\b", str(text or "")):
        cand = _extract_answer_tail_expression(str(text)[m.end() :])
        if cand:
            candidates.append(cand)
    return candidates[-1] if candidates else None


def _canonical_gsm8k_scalar_answer(x: str) -> Optional[str]:
    s = str(x or "").strip()
    if not s or s == "[INVALID]":
        return None

    s = _strip_math_delims(s)
    s = s.replace("−", "-")
    s = re.sub(r"\\[,\;\!\:\ ]\s*", "", s)
    s = s.replace(r"\%", "%")
    s = s.replace("{,}", ",")
    s = s.replace(r"\$", "$")
    s = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r" \1 ", s)
    s = s.replace("$", " ").strip()

    changed = True
    while changed:
        old = s
        s = s.strip()
        s = re.sub(r"^(?:answer|ans)\s*(?:is|=|:)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^(?:about|approximately|approx\.?|around)\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[\.\。;；:：!！?？]+$", "", s).strip()
        if len(s) >= 2 and ((s[0], s[-1]) in {("(", ")"), ("[", "]"), ("{", "}")}):
            s = s[1:-1].strip()
        changed = s != old

    if not s:
        return None
    if re.search(r"\\(?:frac|dfrac|tfrac|sqrt|pi|times|cdot|div)\b", s):
        return None
    if re.search(r"[=+*/^]", s):
        return None
    if re.search(r"(?<!^)-", s):
        return None

    number_pat = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?|[-+]?\.\d+"
    matches = list(re.finditer(number_pat, s))
    if len(matches) != 1:
        return None

    m = matches[0]
    prefix = s[: m.start()].strip()
    raw_suffix = s[m.end() :]
    suffix = raw_suffix.strip()
    if prefix and not re.fullmatch(r"[$￥¥€£]*", prefix):
        return None
    if suffix:
        if raw_suffix and raw_suffix[0].isalpha():
            return None
        if re.search(r"[=+*/^\\{}]", suffix):
            return None
        if re.search(number_pat, suffix):
            return None
        if not re.fullmatch(r"[\sA-Za-z%％$￥¥€£°.,;:()\[\]\-_/]+", suffix):
            return None

    num = m.group(0).replace(",", "")
    try:
        val = float(num)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    if abs(val - round(val)) <= 1e-9:
        return str(int(round(val)))
    return ("%.12f" % val).rstrip("0").rstrip(".")


def _judge_gsm8k(solution: str, gt: str, truncated: bool = False) -> Dict[str, Any]:
    gt_final = _extract_gsm8k_final_answer(gt)
    solution_tail = str(solution or "")[-1200:]
    boxed = _last_boxed(solution_tail)
    if boxed is not None and boxed.strip():
        extracted = boxed
        extract_source = "boxed"
    else:
        marker = _extract_last_valid_answer_marker(solution_tail)
        if marker is not None:
            extracted = marker
            extract_source = "answer_regex"
        else:
            extracted = "[INVALID]"
            extract_source = "invalid"

    pred_scalar = _canonical_gsm8k_scalar_answer(extracted)
    gt_scalar = _canonical_gsm8k_scalar_answer(gt_final)
    ok = bool(pred_scalar is not None and gt_scalar is not None and pred_scalar == gt_scalar)
    if truncated:
        ok = False
    return {
        "ok": ok,
        "gt": gt_final,
        "raw": extracted,
        "extract_source": extract_source,
        "pred_scalar": pred_scalar,
        "gt_scalar": gt_scalar,
        "truncated": bool(truncated),
        "truncated_forced_zero": bool(truncated),
        "error": None,
    }


def _build_gsm8k_prompt(problem: str) -> str:
    p = str(problem or "").strip()
    return (
        f"User: {p}\n"
        f"Solve the problem. Put the final answer in \\boxed{{...}}, and make the final line contain only \\boxed{{...}}. think\n"
        f"Assistant: <think>\n"
    )


@dataclass
class MaxRLConfig:
    train_jsonl: str
    out_dir: str
    model: str
    tokenizer: str
    max_steps: int
    ctx_len: int
    grad_cp: int
    seed: int
    lr: float
    adamw_fallback_lr: float
    beta1: float
    beta2: float
    optimizer_eps: float
    optimizer: str
    muon_momentum: float
    muon_nesterov: bool
    muon_ns_steps: int
    muon_eps: float
    muon_adjust_lr_fn: str
    muon_variant: str
    grad_clip: float
    save_interval: int
    log_interval: int
    micro_batch_size: int
    global_batch_size: int
    memory_efficient_adamw: bool
    max_new_tokens: int
    rollout_batch_size: int
    group_size: int
    temperature: float
    top_p: float
    top_k: int
    use_rapid_sampling: bool
    clip_range: float
    dynamic_prompt_batch: int
    max_sampling_rounds: int
    sync_diag_interval: int
    sync_diag_sample_values: int
    sync_infer_offload_cpu: bool
    eval_jsonl: str
    eval_interval: int
    eval_batch_size: int
    eval_limit: int
    eval_temperature: float
    eval_top_p: float
    eval_top_k: int


# Hand-copied from:
# maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py
def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    del norm_adv_by_std_in_grpo
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack([x.to(scores.device) for x in id2score[idx]])
                id2mean[idx] = torch.mean(score_tensor)
                id2std[idx] = torch.std(score_tensor.unsqueeze(0))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# Hand-copied from:
# grpo_math_local.py::_ppo_clipped_objective
def _ppo_clipped_objective(ratio: torch.Tensor, adv: torch.Tensor, clip_range: float) -> torch.Tensor:
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    return torch.where(adv >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))


def _dist_barrier(train_gpu: int):
    if _dist_is_initialized():
        dist.barrier(device_ids=[int(train_gpu)])


_SYNC_TRANSPOSE_KEYS = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")


def _project_train_tensor_for_infer(name: str, tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach()
    if any(k in name for k in _SYNC_TRANSPOSE_KEYS):
        x = x.t()
    x = x.squeeze()
    if name.endswith("att.r_k"):
        x = x.flatten()
    return x


def _sample_tensor_digest(tensor: Optional[torch.Tensor], sample_values: int) -> Optional[Dict[str, Any]]:
    if not torch.is_tensor(tensor):
        return None
    flat = tensor.reshape(-1)
    if flat.numel() <= 0:
        return {
            "shape": tuple(int(x) for x in tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(flat.numel()),
            "sample_n": 0,
            "sum": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
        }
    n = min(int(sample_values), int(flat.numel()))
    head = flat[:n].float()
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flat.numel()),
        "sample_n": int(n),
        "sum": float(head.sum().item()),
        "abs_mean": float(head.abs().mean().item()),
        "abs_max": float(head.abs().max().item()),
    }


def _sample_tensor_head_cpu(tensor: Optional[torch.Tensor], sample_values: int) -> Optional[torch.Tensor]:
    if not torch.is_tensor(tensor):
        return None
    flat = tensor.reshape(-1)
    if flat.numel() <= 0:
        return torch.empty((0,), dtype=torch.float32)
    n = min(int(sample_values), int(flat.numel()))
    return flat[:n].detach().to(device="cpu", dtype=torch.float32)


def _select_sync_diag_names(base_model: torch.nn.Module, infer_engine, n_layers: int) -> List[str]:
    if infer_engine is None or not hasattr(infer_engine, "infer_model") or not hasattr(infer_engine.infer_model, "z"):
        return []
    infer_keys = set(getattr(infer_engine.infer_model, "z", {}).keys())
    last = max(0, int(n_layers) - 1)
    mid = max(0, int(n_layers) // 2)
    preferred = [
        "head.weight",
        "blocks.0.att.key.weight",
        "blocks.0.att.value.weight",
        "blocks.0.ffn.key.weight",
        f"blocks.{mid}.att.key.weight",
        f"blocks.{mid}.ffn.value.weight",
        f"blocks.{last}.att.output.weight",
        f"blocks.{last}.ffn.receptance.weight",
    ]
    picked: List[str] = []
    for name in preferred:
        if name in infer_keys and _get_tensor_by_dotted_name(base_model, name) is not None and name not in picked:
            picked.append(name)
    if picked:
        return picked[:6]
    fallback = []
    for name in sorted(infer_keys):
        if name.startswith("emb.") or name.endswith("ln0.weight") or name.endswith("ln0.bias"):
            continue
        if _get_tensor_by_dotted_name(base_model, name) is not None:
            fallback.append(name)
        if len(fallback) >= 6:
            break
    return fallback


def _run_sync_diagnostics(
    *,
    step: int,
    base_model: torch.nn.Module,
    infer_engine,
    diag_names: List[str],
    sample_values: int,
    world_size: int,
    logger: sft_backend.Logger,
    is_main: bool,
):
    if not diag_names or infer_engine is None:
        return

    local_payload: Dict[str, Any] = {}
    infer_z = getattr(getattr(infer_engine, "infer_model", None), "z", {})
    for name in diag_names:
        src = _get_tensor_by_dotted_name(base_model, name)
        infer_t = infer_z.get(name) if isinstance(infer_z, dict) else None
        if not torch.is_tensor(src):
            local_payload[name] = {"missing_train": True}
            continue
        proj = _project_train_tensor_for_infer(name, src)
        proj_fp16 = proj.to(dtype=torch.half)
        proj_digest = _sample_tensor_digest(proj_fp16, sample_values=sample_values)
        infer_digest = _sample_tensor_digest(infer_t, sample_values=sample_values)
        item: Dict[str, Any] = {
            "train_proj": proj_digest,
            "infer": infer_digest,
        }
        if proj_digest is not None and infer_digest is not None and proj_digest["sample_n"] > 0 and infer_digest["sample_n"] > 0:
            n = min(int(proj_digest["sample_n"]), int(infer_digest["sample_n"]))
            proj_head = _sample_tensor_head_cpu(proj_fp16, sample_values=n)
            infer_head = _sample_tensor_head_cpu(infer_t, sample_values=n)
            if proj_head is None or infer_head is None:
                local_payload[name] = item
                continue
            diff = (infer_head - proj_head).abs()
            item["infer_diff_max"] = float(diff.max().item())
            item["infer_diff_mean"] = float(diff.mean().item())
        local_payload[name] = item

    gathered = [None for _ in range(world_size)] if world_size > 1 and _dist_is_initialized() else [local_payload]
    if world_size > 1 and _dist_is_initialized():
        dist.all_gather_object(gathered, local_payload)

    if not is_main:
        return

    lines = [f"[sync-diag step {step}] sampled_names={len(diag_names)} sample_values={int(sample_values)}"]
    for name in diag_names:
        rank_entries = []
        for rank_idx, payload in enumerate(gathered):
            if not isinstance(payload, dict):
                continue
            item = payload.get(name)
            if isinstance(item, dict):
                rank_entries.append((rank_idx, item))
        if not rank_entries:
            lines.append(f"  {name}: missing on all ranks")
            continue

        base_item = rank_entries[0][1]
        base_train = base_item.get("train_proj") if isinstance(base_item, dict) else None
        rank_mismatch = False
        infer_bad = False
        parts = []
        for rank_idx, item in rank_entries:
            train_proj = item.get("train_proj")
            infer_diff_max = float(item.get("infer_diff_max", float("nan")))
            infer_diff_mean = float(item.get("infer_diff_mean", float("nan")))
            train_sum = float("nan")
            train_absmax = float("nan")
            if isinstance(base_train, dict) and isinstance(train_proj, dict):
                if (
                    tuple(train_proj.get("shape", ())) != tuple(base_train.get("shape", ()))
                    or abs(float(train_proj.get("sum", 0.0)) - float(base_train.get("sum", 0.0))) > 1e-3
                    or abs(float(train_proj.get("abs_max", 0.0)) - float(base_train.get("abs_max", 0.0))) > 1e-3
                ):
                    rank_mismatch = True
            if isinstance(train_proj, dict):
                train_sum = float(train_proj.get("sum", float("nan")))
                train_absmax = float(train_proj.get("abs_max", float("nan")))
            if math.isfinite(infer_diff_max) and infer_diff_max > 1e-6:
                infer_bad = True
            parts.append(
                f"r{rank_idx}:train_sum={train_sum:.4e} "
                f"train_absmax={train_absmax:.4e} "
                f"infer_diff_max={infer_diff_max:.4e} infer_diff_mean={infer_diff_mean:.4e}"
            )
        flag = "WARN" if (rank_mismatch or infer_bad) else "OK"
        lines.append(f"  [{flag}] {name} | " + " | ".join(parts))

    for line in lines:
        logger.log(line)


# Hand-copied/adapted from:
# grpo_math_local.py::_sync_infer_weights
# WARNING:
# The user explicitly noted that the FSDP train->rollout sync path in
# grpo_math_local.py may contain a severe bug. This function is still copied
# first for fidelity, but should be treated as suspicious until validated.
# In particular, `sync_infer_offload_cpu` is added here to match the original
# grpo_math_local.py default, but this offload path is still UNVERIFIED in this
# script and may contain serious hazards.
@torch.no_grad()
def _sync_infer_weights(
    *,
    step: int,
    train_model: torch.nn.Module,
    base_model: torch.nn.Module,
    infer_engine,
    fsdp_enabled: bool,
    world_size: int,
    train_gpu: int,
    logger: sft_backend.Logger,
    is_main: bool,
    sync_diag_interval: int,
    sync_diag_sample_values: int,
    sync_infer_offload_cpu: bool,
):
    if infer_engine is None:
        return

    if fsdp_enabled and world_size > 1:
        do_sync = True
        if hasattr(infer_engine, "should_sync"):
            do_sync = bool(infer_engine.should_sync(step=step, force=True))
        if _dist_is_initialized():
            flag = [1 if do_sync else 0]
            dist.broadcast_object_list(flag, src=0)
            do_sync = bool(flag[0])
        if not do_sync:
            _dist_barrier(train_gpu)
            return

        if FSDP is None:
            raise RuntimeError("FSDP state-dict helpers are unavailable in current torch build.")

        n_layers = int(getattr(getattr(base_model, "args", None), "n_layer", 0))
        t_sync = time.time()
        diag_names = []
        if int(sync_diag_interval) > 0 and (int(step) == 1 or int(step) % int(sync_diag_interval) == 0):
            diag_names = _select_sync_diag_names(base_model, infer_engine, n_layers=n_layers)

        try:
            ctx = FSDP.summon_full_params(
                train_model,
                recurse=True,
                writeback=False,
                rank0_only=False,
                offload_to_cpu=bool(sync_infer_offload_cpu),
            )
        except TypeError:
            ctx = FSDP.summon_full_params(train_model, recurse=True, writeback=False)

        def _getter(name: str):
            return _get_tensor_by_dotted_name(base_model, name)

        with ctx:
            infer_engine.sync_infer_weights(
                step=step,
                force=True,
                train_tensor_getter=_getter,
                n_layers=n_layers,
            )
            if diag_names:
                _run_sync_diagnostics(
                    step=step,
                    base_model=base_model,
                    infer_engine=infer_engine,
                    diag_names=diag_names,
                    sample_values=int(sync_diag_sample_values),
                    world_size=world_size,
                    logger=logger,
                    is_main=is_main,
                )
        if is_main:
            logger.log(
                f"[sync] train->rollout done: step={step} "
                f"offload_cpu={int(bool(sync_infer_offload_cpu))} dt={time.time() - t_sync:.2f}s"
            )
        _dist_barrier(train_gpu)
        return

    infer_engine.sync_infer_weights(step=step, force=True)


def _normalize_train_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        problem = str(rec.get("problem") or rec.get("question") or rec.get("prompt") or rec.get("input") or "").strip()
        answer_src = str(rec.get("answer") or rec.get("solution") or rec.get("output") or rec.get("target") or "").strip()
        answer = _extract_gsm8k_final_answer(answer_src)
        if not problem or not answer:
            continue
        normalized.append(
            {
                "problem": problem,
                "answer": answer,
            }
        )
    return normalized


def _split_global_count(total: int, world_size: int, rank: int) -> Tuple[int, int, int]:
    base = int(total) // max(1, int(world_size))
    rem = int(total) % max(1, int(world_size))
    start = int(rank) * base + min(int(rank), rem)
    local = base + (1 if int(rank) < rem else 0)
    end = start + local
    return local, start, end


def _group_is_all_zero_or_all_one(group: Dict[str, Any]) -> bool:
    rewards = [float(sample.get("reward", 0.0)) for sample in group.get("samples", [])]
    if not rewards:
        return True
    all_zero = all(r == 0.0 for r in rewards)
    all_one = all(r == 1.0 for r in rewards)
    return all_zero or all_one


def _group_reward_counts(groups: List[Dict[str, Any]]) -> Tuple[int, int, int, int, int]:
    sample_n = 0
    correct_n = 0
    all0_n = 0
    all1_n = 0
    mixed_n = 0
    for group in groups:
        rewards = [float(sample.get("reward", 0.0)) for sample in group.get("samples", [])]
        if not rewards:
            all0_n += 1
            continue
        ok_n = sum(1 for r in rewards if r == 1.0)
        sample_n += len(rewards)
        correct_n += int(ok_n)
        if ok_n == 0:
            all0_n += 1
        elif ok_n == len(rewards):
            all1_n += 1
        else:
            mixed_n += 1
    return sample_n, correct_n, all0_n, all1_n, mixed_n


def _build_prompt_ids(record: Dict[str, Any], tokenizer, ctx_len: int, max_new_tokens: int) -> List[int]:
    prompt = _build_gsm8k_prompt(record["problem"])
    ids = tokenizer.encode(prompt)
    max_prompt_len = int(ctx_len) - int(max_new_tokens) - 4
    max_prompt_len = max(64, max_prompt_len)
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]
    return [int(x) for x in ids]


@torch.no_grad()
def _logprob_delta_stats_local(
    *,
    train_model,
    prompt_groups: List[Dict[str, Any]],
    device: str,
    clip_range: float,
    tokenizer=None,
) -> Dict[str, float]:
    delta_tok_sum = 0.0
    delta_tok_abs_sum = 0.0
    delta_seq_sum = 0.0
    delta_seq_abs_sum = 0.0
    delta_tok_cnt = 0.0
    delta_seq_cnt = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    clip_cnt = 0.0
    seq_delta_min = float("inf")
    seq_delta_max = float("-inf")
    seq_delta_abs_max = 0.0
    top_pos_delta: Optional[Dict[str, Any]] = None
    top_neg_delta: Optional[Dict[str, Any]] = None
    top_abs_delta: Optional[Dict[str, Any]] = None
    top_pos_seq: Optional[Dict[str, Any]] = None
    top_neg_seq: Optional[Dict[str, Any]] = None
    top_abs_seq: Optional[Dict[str, Any]] = None

    def _decode_ids(ids: List[int]) -> str:
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode([int(x) for x in ids], utf8_errors="replace")
        except TypeError:
            pass
        except Exception:
            return ""
        try:
            return tokenizer.decode([int(x) for x in ids])
        except Exception:
            return ""

    def _tok_record(
        *,
        group: Dict[str, Any],
        sample: Dict[str, Any],
        offset: int,
        token_id: int,
        old_lp: float,
        new_lp: float,
        delta_v: float,
        seq_delta_v: float,
    ) -> Dict[str, Any]:
        comp_tokens = [int(x) for x in sample.get("comp_tokens", [])]
        lo = max(0, int(offset) - 12)
        hi = min(len(comp_tokens), int(offset) + 13)
        return {
            "problem": str(group.get("problem", ""))[:240],
            "answer": str(group.get("answer", ""))[:80],
            "offset": int(offset),
            "token_id": int(token_id),
            "token_text": _decode_ids([int(token_id)]),
            "old_lp": float(old_lp),
            "new_lp": float(new_lp),
            "delta": float(delta_v),
            "token_ratio": float(math.exp(max(-50.0, min(50.0, float(delta_v))))),
            "seq_delta": float(seq_delta_v),
            "seq_ratio": float(math.exp(max(-50.0, min(50.0, float(seq_delta_v))))),
            "sample_reward": float(sample.get("reward", 0.0)),
            "sample_adv": float(sample.get("adv", 0.0)),
            "sample_truncated": bool(sample.get("truncated", False)),
            "completion_prefix": str(sample.get("completion", ""))[:500],
            "token_window_ids": comp_tokens[lo:hi],
            "token_window_text": _decode_ids(comp_tokens[lo:hi]),
        }

    def _seq_record(group: Dict[str, Any], sample: Dict[str, Any], seq_delta_v: float, tok_n: int) -> Dict[str, Any]:
        return {
            "problem": str(group.get("problem", ""))[:240],
            "answer": str(group.get("answer", ""))[:80],
            "seq_delta": float(seq_delta_v),
            "seq_ratio": float(math.exp(max(-50.0, min(50.0, float(seq_delta_v))))),
            "tok_n": int(tok_n),
            "sample_reward": float(sample.get("reward", 0.0)),
            "sample_adv": float(sample.get("adv", 0.0)),
            "sample_truncated": bool(sample.get("truncated", False)),
            "completion_prefix": str(sample.get("completion", ""))[:1000],
        }

    for group in prompt_groups:
        prompt_ids = [int(x) for x in group.get("prompt_ids", [])]
        prompt_len = len(prompt_ids)
        if prompt_len <= 0:
            continue
        for sample in group.get("samples", []):
            keep = min(len(sample.get("comp_tokens", [])), len(sample.get("old_logps", [])))
            if keep <= 0:
                continue
            full_tokens = prompt_ids + [int(x) for x in sample["comp_tokens"][:keep]]
            padded, lens = _pad_batch_local([full_tokens], device=device, pad_id=0)
            inp = padded[:, :-1].contiguous()
            tgt = padded[:, 1:].contiguous()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = train_model(inp)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits.unsqueeze(0)
            picked = -F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            ).reshape_as(tgt)

            full_len = int(lens[0])
            start = max(0, prompt_len - 1)
            end = max(start, min(full_len - 1, start + keep))
            if end <= start:
                continue

            new_lp = picked[0, start:end].float()
            old_lp = torch.tensor(sample["old_logps"][:keep], dtype=torch.float32, device=device)
            if old_lp.numel() != new_lp.numel():
                take = min(old_lp.numel(), new_lp.numel())
                if take <= 0:
                    continue
                new_lp = new_lp[:take]
                old_lp = old_lp[:take]

            delta = new_lp - old_lp
            ratio = torch.exp(delta.clamp(min=-20.0, max=20.0))
            tok_n = int(delta.numel())
            if tok_n <= 0:
                continue
            seq_delta = float(delta.sum().item())
            delta_tok_sum += float(delta.sum().item())
            delta_tok_abs_sum += float(delta.abs().sum().item())
            delta_seq_sum += seq_delta
            delta_seq_abs_sum += abs(seq_delta)
            delta_tok_cnt += float(tok_n)
            delta_seq_cnt += 1.0
            ratio_sum += float(ratio.sum().item())
            ratio_max = max(ratio_max, float(ratio.max().item()))
            seq_delta_min = min(seq_delta_min, seq_delta)
            seq_delta_max = max(seq_delta_max, seq_delta)
            seq_delta_abs_max = max(seq_delta_abs_max, abs(seq_delta))
            seq_rec = _seq_record(group, sample, seq_delta, tok_n)
            if top_pos_seq is None or seq_delta > float(top_pos_seq["seq_delta"]):
                top_pos_seq = seq_rec
            if top_neg_seq is None or seq_delta < float(top_neg_seq["seq_delta"]):
                top_neg_seq = seq_rec
            if top_abs_seq is None or abs(seq_delta) > abs(float(top_abs_seq["seq_delta"])):
                top_abs_seq = seq_rec

            delta_cpu = delta.detach().float().cpu()
            old_cpu = old_lp.detach().float().cpu()
            new_cpu = new_lp.detach().float().cpu()
            comp_tokens = [int(x) for x in sample.get("comp_tokens", [])]
            if int(delta_cpu.numel()) > 0:
                pos_i = int(torch.argmax(delta_cpu).item())
                neg_i = int(torch.argmin(delta_cpu).item())
                abs_i = int(torch.argmax(delta_cpu.abs()).item())
                candidates = [
                    ("pos", pos_i, float(delta_cpu[pos_i].item())),
                    ("neg", neg_i, float(delta_cpu[neg_i].item())),
                    ("abs", abs_i, float(delta_cpu[abs_i].item())),
                ]
                for kind, idx_i, delta_i in candidates:
                    token_id = comp_tokens[idx_i] if idx_i < len(comp_tokens) else -1
                    rec = _tok_record(
                        group=group,
                        sample=sample,
                        offset=idx_i,
                        token_id=token_id,
                        old_lp=float(old_cpu[idx_i].item()),
                        new_lp=float(new_cpu[idx_i].item()),
                        delta_v=delta_i,
                        seq_delta_v=seq_delta,
                    )
                    if kind == "pos" and (top_pos_delta is None or delta_i > float(top_pos_delta["delta"])):
                        top_pos_delta = rec
                    elif kind == "neg" and (top_neg_delta is None or delta_i < float(top_neg_delta["delta"])):
                        top_neg_delta = rec
                    elif kind == "abs" and (top_abs_delta is None or abs(delta_i) > abs(float(top_abs_delta["delta"]))):
                        top_abs_delta = rec
            clip_cnt += float(
                ((ratio > (1.0 + float(clip_range))) | (ratio < (1.0 - float(clip_range))))
                .float()
                .sum()
                .item()
            )

    return {
        "delta_tok_sum": delta_tok_sum,
        "delta_tok_abs_sum": delta_tok_abs_sum,
        "delta_seq_sum": delta_seq_sum,
        "delta_seq_abs_sum": delta_seq_abs_sum,
        "delta_tok_cnt": delta_tok_cnt,
        "delta_seq_cnt": delta_seq_cnt,
        "ratio_sum": ratio_sum,
        "ratio_max": ratio_max,
        "clip_cnt": clip_cnt,
        "seq_delta_min": 0.0 if delta_seq_cnt <= 0 else seq_delta_min,
        "seq_delta_max": 0.0 if delta_seq_cnt <= 0 else seq_delta_max,
        "seq_delta_abs_max": seq_delta_abs_max,
        "seq_ratio_max": 0.0 if delta_seq_cnt <= 0 else math.exp(max(-50.0, min(50.0, seq_delta_max))),
        "top_pos_delta": top_pos_delta,
        "top_neg_delta": top_neg_delta,
        "top_abs_delta": top_abs_delta,
        "top_pos_seq": top_pos_seq,
        "top_neg_seq": top_neg_seq,
        "top_abs_seq": top_abs_seq,
    }


def _append_jsonl(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def _evaluate_math(
    *,
    step: int,
    eval_data: List[Dict[str, Any]],
    infer_engine,
    tokenizer,
    cfg: MaxRLConfig,
    device: str,
    rank: int,
    world_size: int,
    train_gpu: int,
    logger: sft_backend.Logger,
    is_main: bool,
):
    if infer_engine is None:
        raise RuntimeError("Evaluation requires infer_engine, but it is unavailable.")
    if not eval_data:
        raise RuntimeError("Evaluation dataset is empty.")

    idxs = list(range(len(eval_data)))
    if world_size > 1:
        local_pos = list(range(rank, len(idxs), world_size))
        local_idxs = [idxs[i] for i in local_pos]
    else:
        local_idxs = list(idxs)

    ex_list = [eval_data[i] for i in local_idxs]
    prompt_tokens_list = [_build_prompt_ids(ex, tokenizer, cfg.ctx_len, cfg.max_new_tokens) for ex in ex_list]
    gts = [str(ex.get("answer", ex.get("solution", ""))).strip() for ex in ex_list]

    comp_tokens: List[List[int]] = []
    comp_texts: List[str] = []
    truncated: List[bool] = []
    for start in range(0, len(prompt_tokens_list), int(cfg.eval_batch_size)):
        batch_prompts = prompt_tokens_list[start : start + int(cfg.eval_batch_size)]
        batch_seed = int(cfg.seed + step * 1009 + start * 17 + rank * 1000003 + 7)
        batch_tokens, _, batch_texts, batch_trunc = infer_engine.generate_group_parallel(
            prompt_tokens_list=batch_prompts,
            group_size=1,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.eval_temperature),
            top_p=float(cfg.eval_top_p),
            top_k=int(cfg.eval_top_k),
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=False,
            stop_check_every=8,
            stop_check_window=96,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            penalty_decay=0.0,
            use_rollout_cache=False,
            rng_seed=batch_seed,
        )
        comp_tokens.extend(batch_tokens)
        comp_texts.extend(batch_texts)
        truncated.extend(batch_trunc)

    local_correct = 0
    local_trunc = 0
    local_len_sum = 0.0
    details: List[Dict[str, Any]] = []
    for i, ex in enumerate(ex_list):
        judge = _judge_gsm8k(comp_texts[i], gts[i], bool(truncated[i]))
        ok = bool(judge.get("ok", False))
        local_correct += int(ok)
        local_trunc += int(bool(truncated[i]))
        local_len_sum += float(len(comp_tokens[i]))
        details.append(
            {
                "idx": int(ex.get("_eval_source_idx", local_idxs[i])),
                "local_eval_idx": int(local_idxs[i]),
                "problem": ex.get("problem", ""),
                "gt": gts[i],
                "completion": comp_texts[i],
                "truncated": bool(truncated[i]),
                "judge": judge,
                "reward": 1.0 if ok else 0.0,
                "gen_len": len(comp_tokens[i]),
            }
        )

    stats = torch.tensor(
        [
            float(local_correct),
            float(local_trunc),
            float(local_len_sum),
            float(len(ex_list)),
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered_details = [None for _ in range(world_size)] if world_size > 1 and _dist_is_initialized() else [details]
    if world_size > 1 and _dist_is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_gather_object(gathered_details, details)

    if not is_main:
        if world_size > 1 and _dist_is_initialized():
            _dist_barrier(train_gpu)
        return

    total_correct = int(stats[0].item())
    total_trunc = int(stats[1].item())
    total_len_sum = float(stats[2].item())
    total_n = int(stats[3].item())
    merged_details: List[Dict[str, Any]] = []
    for rank_details in gathered_details:
        if isinstance(rank_details, list):
            merged_details.extend(rank_details)
    merged_details.sort(key=lambda x: int(x.get("idx", -1)))

    eval_summary = {
        "time": now_str(),
        "step": int(step),
        "eval_n": int(total_n),
        "judge_acc": float(total_correct / max(1, total_n)),
        "trunc_rate": float(total_trunc / max(1, total_n)),
        "avg_len": float(total_len_sum / max(1, total_n)),
        "eval_temperature": float(cfg.eval_temperature),
        "eval_top_p": float(cfg.eval_top_p),
        "eval_top_k": int(cfg.eval_top_k),
        "eval_max_new_tokens": int(cfg.max_new_tokens),
        "group_size": 1,
        "world_size": int(world_size),
        "eval_jsonl": str(cfg.eval_jsonl),
    }
    eval_outputs = dict(eval_summary)
    eval_outputs["details"] = merged_details
    _append_jsonl(os.path.join(cfg.out_dir, "eval.jsonl"), eval_summary)
    _append_jsonl(os.path.join(cfg.out_dir, "eval_gen_judgements.jsonl"), eval_outputs)
    logger.log(
        f"[EVAL step {step}] judge_acc={eval_summary['judge_acc']:.3f} "
        f"n={total_n} trunc={eval_summary['trunc_rate']:.3f} avg_len={eval_summary['avg_len']:.1f}"
    )
    if world_size > 1 and _dist_is_initialized():
        _dist_barrier(train_gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/rwkv7-g1g-1.5b-20260526-ctx8192.pth")
    ap.add_argument("--train_jsonl", type=str, default="gsmk8ktrain.parquet")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--adamw_fallback_lr", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--optimizer_eps", type=float, default=1e-18)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "muon"])
    ap.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="auto",
        choices=["auto", "full_shard", "shard_grad_op", "no_shard"],
        help="FSDP sharding. auto uses shard_grad_op for Muon so 2D params remain visible.",
    )
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--muon_nesterov", dest="muon_nesterov", action="store_true")
    ap.add_argument("--no_muon_nesterov", dest="muon_nesterov", action="store_false")
    ap.set_defaults(muon_nesterov=True)
    ap.add_argument("--muon_ns_steps", type=int, default=5)
    ap.add_argument("--muon_eps", type=float, default=1e-7)
    ap.add_argument("--muon_adjust_lr_fn", type=str, default="match_rms_adamw", choices=["original", "match_rms_adamw", "none"])
    ap.add_argument("--muon_variant", type=str, default="torch", choices=["torch", "moonlight"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--global_batch_size", type=int, default=16)
    ap.add_argument("--memory_efficient_adamw", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--rollout_batch_size", type=int, default=128)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--use_rapid_sampling", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--sync_diag_interval", type=int, default=50)
    ap.add_argument("--sync_diag_sample_values", type=int, default=1024)
    ap.add_argument("--sync_infer_offload_cpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_jsonl", type=str, default="gsmk_test.json")
    ap.add_argument("--eval_interval", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--eval_limit", type=int, default=500)
    ap.add_argument("--eval_temperature", type=float, default=0.3)
    ap.add_argument("--eval_top_p", type=float, default=0.4)
    ap.add_argument("--eval_top_k", type=int, default=20)
    ap.add_argument("--overfit_one", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument(
        "--no-fsdp_no_sync",
        dest="fsdp_no_sync",
        action="store_false",
        default=False,
        help="FSDP no_sync is forcibly disabled; this flag is kept only for CLI compatibility.",
    )
    if any(
        x == "--fsdp_no_sync"
        or x.startswith("--fsdp_no_sync=")
        or x == "--fsdp-no-sync"
        or x.startswith("--fsdp-no-sync=")
        for x in sys.argv[1:]
    ):
        raise SystemExit("Parameter forbidden: fsdp_no_sync is hard-disabled and must remain False.")
    args = ap.parse_args()

    if int(args.micro_batch_size) != 1:
        raise RuntimeError("This MaxRL entry currently expects --micro_batch_size=1 to match the SFT backend.")

    fixed_rollout = {
        "ctx_len": 8192,
        "max_new_tokens": 512,
        "rollout_batch_size": 128,
        "group_size": 8,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "tokenizer": "reference/rwkv_vocab_v20230424.txt",
        "seed": 42,
        "use_rapid_sampling": True,
    }
    mismatches = []
    for key, expected in fixed_rollout.items():
        actual = getattr(args, key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise SystemExit("Rollout parameter mismatch: " + ", ".join(mismatches))

    rank, world_size, local_rank = _init_distributed_from_env()
    is_main = rank == 0

    def rank0_print(msg: str):
        if is_main:
            print(msg, flush=True)

    auto_out_dir = False
    if args.out_dir is None or str(args.out_dir).strip() == "":
        auto_out_dir = True
        if is_main:
            args.out_dir = f"out_maxrl_gsm8k_{now_str()}"
    if world_size > 1 and _dist_is_initialized() and auto_out_dir:
        out_obj = [args.out_dir if is_main else None]
        dist.broadcast_object_list(out_obj, src=0)
        args.out_dir = str(out_obj[0])

    os.makedirs(args.out_dir, exist_ok=True)
    logger = sft_backend.Logger(args.out_dir, rank=rank, is_main=is_main)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    n_cuda = torch.cuda.device_count()
    if n_cuda < 1:
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

    available_gpu_indices = list(range(n_cuda))
    if world_size > len(available_gpu_indices):
        raise RuntimeError(
            f"WORLD_SIZE={world_size} exceeds available training GPUs={len(available_gpu_indices)} "
            f"(train_gpus={available_gpu_indices})."
        )

    active_train_gpus = available_gpu_indices[: max(1, world_size)]
    if world_size > 1:
        if local_rank >= world_size:
            raise RuntimeError(f"LOCAL_RANK={local_rank} out of range for WORLD_SIZE={world_size}.")
        train_gpu = int(local_rank)
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        rank0_print(
            f"[GPU] distributed mode: world_size={world_size} "
            f"train_gpus={active_train_gpus} (rank{rank} -> cuda:{train_gpu})"
        )
    else:
        train_gpu = active_train_gpus[0]
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        rank0_print(f"[GPU] single-process mode: total={n_cuda}, train_gpu={train_gpu}")

    os.environ["RWKV_HEAD_SIZE_A"] = "64"
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    _ensure_cuda_toolkit_env()
    safe_arch = _resolve_torch_cuda_arch_list(device)
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if safe_arch:
        if arch_env and "10.0" in arch_env:
            os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch
            rank0_print(f"[CUDA-ARCH] override TORCH_CUDA_ARCH_LIST={arch_env} -> {safe_arch}")
        elif not arch_env:
            os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch
            rank0_print(f"[CUDA-ARCH] set TORCH_CUDA_ARCH_LIST={safe_arch}")

    cfg = MaxRLConfig(
        train_jsonl=str(args.train_jsonl),
        out_dir=str(args.out_dir),
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        max_steps=int(args.max_steps),
        ctx_len=int(args.ctx_len),
        grad_cp=int(args.grad_cp),
        seed=int(args.seed),
        lr=float(args.lr),
        adamw_fallback_lr=float(args.adamw_fallback_lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        optimizer_eps=float(args.optimizer_eps),
        optimizer=str(args.optimizer),
        muon_momentum=float(args.muon_momentum),
        muon_nesterov=bool(args.muon_nesterov),
        muon_ns_steps=int(args.muon_ns_steps),
        muon_eps=float(args.muon_eps),
        muon_adjust_lr_fn=str(args.muon_adjust_lr_fn),
        muon_variant=str(args.muon_variant),
        grad_clip=float(args.grad_clip),
        save_interval=int(args.save_interval),
        log_interval=int(args.log_interval),
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
        memory_efficient_adamw=bool(args.memory_efficient_adamw),
        max_new_tokens=int(args.max_new_tokens),
        rollout_batch_size=int(args.rollout_batch_size),
        group_size=int(args.group_size),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        use_rapid_sampling=bool(args.use_rapid_sampling),
        clip_range=float(args.clip_range),
        dynamic_prompt_batch=24,
        max_sampling_rounds=10,
        sync_diag_interval=int(args.sync_diag_interval),
        sync_diag_sample_values=int(args.sync_diag_sample_values),
        sync_infer_offload_cpu=bool(args.sync_infer_offload_cpu),
        eval_jsonl=str(args.eval_jsonl),
        eval_interval=int(args.eval_interval),
        eval_batch_size=int(args.eval_batch_size),
        eval_limit=int(args.eval_limit),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
    )
    if cfg.global_batch_size <= 0:
        raise RuntimeError("--global_batch_size must be > 0")
    if cfg.global_batch_size % max(1, world_size) != 0:
        raise RuntimeError(
            f"--global_batch_size={cfg.global_batch_size} must be divisible by world_size={world_size}."
        )
    local_update_group_target = cfg.global_batch_size // max(1, world_size)
    if local_update_group_target <= 0:
        raise RuntimeError("local update group target resolved to <= 0")

    if is_main:
        with open(os.path.join(args.out_dir, "maxrl_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    data_raw = load_data(args.train_jsonl)
    data = _normalize_train_records(data_raw)
    if not data:
        raise RuntimeError("No valid train records after preprocessing.")
    eval_raw = load_data(args.eval_jsonl)
    eval_data_all = _normalize_train_records(eval_raw)
    eval_limit = int(cfg.eval_limit)
    if eval_limit > 0 and len(eval_data_all) > eval_limit:
        eval_rng = random.Random(int(cfg.seed) + 500)
        eval_selected_idxs = sorted(eval_rng.sample(range(len(eval_data_all)), eval_limit))
        eval_data = [dict(eval_data_all[i], _eval_source_idx=int(i)) for i in eval_selected_idxs]
    else:
        eval_selected_idxs = list(range(len(eval_data_all)))
        eval_data = [dict(x, _eval_source_idx=int(i)) for i, x in enumerate(eval_data_all)]
    if not eval_data:
        raise RuntimeError("No valid eval records after preprocessing.")
    data_stats = {
        "loaded_records": len(data_raw),
        "valid_records": len(data),
        "eval_loaded_records": len(eval_raw),
        "eval_valid_records": len(eval_data_all),
        "eval_selected_records": len(eval_data),
        "eval_limit": int(eval_limit),
        "eval_selected_idxs": eval_selected_idxs,
        "first_problem_preview": data[0]["problem"][:200],
        "first_answer_preview": data[0]["answer"][:200],
    }
    if is_main:
        with open(os.path.join(args.out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
            json.dump(data_stats, f, ensure_ascii=False, indent=2)

    overfit_record: Optional[Dict[str, Any]] = None
    logger.log(
        f"Data loaded: valid={data_stats['valid_records']}/{data_stats['loaded_records']} "
        f"eval={data_stats['eval_valid_records']}/{data_stats['eval_loaded_records']} "
        f"target_valid_groups={cfg.global_batch_size} local_update_group_target={local_update_group_target} "
        f"group_size={cfg.group_size} overfit_one={int(bool(args.overfit_one))}"
    )
    if bool(args.overfit_one):
        logger.log("[OVERFIT] will select the first non-degenerate rollout question, then repeat it.")

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    train_idx = _parse_cuda_index(device)
    if train_idx is not None:
        torch.cuda.set_device(train_idx)
    train_model_raw, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device=device,
        ctx_len=int(args.ctx_len),
        grad_cp=int(args.grad_cp),
    )
    if train_idx is not None:
        torch.cuda.set_device(train_idx)
    trainable = enable_full_finetune(train_model_raw)
    if trainable <= 0:
        raise RuntimeError("No trainable parameters found.")
    rank0_print(f"Trainable parameters (full): {trainable}")

    train_model = train_model_raw
    if world_size > 1:
        if FSDP is None or ShardingStrategy is None or MixedPrecision is None or transformer_auto_wrap_policy is None:
            raise RuntimeError("Current torch build does not provide FSDP.")
        fsdp_force_fp32 = os.environ.get("RWKV_FSDP_FORCE_FP32", "0") == "1"
        fsdp_enable_mp = os.environ.get("RWKV_FSDP_ENABLE_MP", "0") == "1"
        if fsdp_force_fp32:
            fsdp_mp = None
            rank0_print("[GPU] FSDP mixed_precision disabled by RWKV_FSDP_FORCE_FP32=1")
        elif fsdp_enable_mp:
            fsdp_mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            rank0_print("[GPU] FSDP mixed_precision enabled by RWKV_FSDP_ENABLE_MP=1")
        else:
            fsdp_mp = None
            rank0_print("[GPU] FSDP mixed_precision disabled by default (match sft_tool_use_fsdp.py)")
        auto_wrap_policy = None
        if hasattr(train_model_raw, "blocks") and len(getattr(train_model_raw, "blocks", [])) > 0:
            block_cls = type(train_model_raw.blocks[0])
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={block_cls},
            )
        sharding_arg = str(args.fsdp_sharding_strategy).lower()
        if sharding_arg == "auto":
            sharding_arg = "shard_grad_op" if str(args.optimizer).lower() == "muon" else "full_shard"
        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        fsdp_sharding = sharding_map[sharding_arg]
        rank0_print(f"[GPU] FSDP sharding_strategy={sharding_arg} optimizer={args.optimizer}")
        train_model = FSDP(
            train_model_raw,
            device_id=torch.cuda.current_device(),
            sharding_strategy=fsdp_sharding,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
            mixed_precision=fsdp_mp,
            auto_wrap_policy=auto_wrap_policy,
        )
        rank0_print(f"[GPU] FSDP enabled across {world_size} training ranks.")

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable params found after wrapping.")

    if str(args.optimizer).lower() == "muon":
        named_trainable = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
        opt = Muon2DWithAdamWFallback(
            named_trainable,
            lr=float(args.lr),
            adam_lr=float(args.adamw_fallback_lr),
            betas=(float(args.beta1), float(args.beta2)),
            adam_eps=float(args.optimizer_eps),
            muon_momentum=float(args.muon_momentum),
            muon_nesterov=bool(args.muon_nesterov),
            muon_ns_steps=int(args.muon_ns_steps),
            muon_eps=float(args.muon_eps),
            muon_adjust_lr_fn=str(args.muon_adjust_lr_fn),
            muon_variant=str(args.muon_variant),
            muon_weight_decay=0.0,
            adam_weight_decay=0.0,
        )
        muon_n = sum(p.numel() for p in opt.muon_params)
        adam_n = sum(p.numel() for p in opt.adam_params)
        logger.log(
            f"Optimizer: Muon2DWithAdamWFallback "
            f"(muon_tensors={len(opt.muon_params)} muon_params={muon_n} "
            f"adam_tensors={len(opt.adam_params)} adam_params={adam_n} "
            f"muon_lr={float(args.lr)} adam_lr={opt.adam_lr} "
            f"betas=({float(args.beta1)},{float(args.beta2)}) adam_eps={float(args.optimizer_eps)} "
            f"muon_eps={float(args.muon_eps)} grad_clip={float(args.grad_clip)} "
            f"variant={opt.muon_variant} momentum={float(args.muon_momentum)} nesterov={bool(args.muon_nesterov)} "
            f"ns_steps={int(args.muon_ns_steps)} adjust={str(args.muon_adjust_lr_fn)} weight_decay=0)"
        )
    elif str(args.optimizer).lower() == "adamw" and args.memory_efficient_adamw:
        opt = MemoryEfficientAdamW(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
            enabled=True,
        )
        logger.log(
            f"Optimizer: MemoryEfficientAdamW (CPU-offloaded states, lr={float(args.lr)} "
            f"betas=({float(args.beta1)},{float(args.beta2)}) eps={float(args.optimizer_eps)} "
            f"grad_clip={float(args.grad_clip)} weight_decay=0)"
        )
    else:
        opt = torch.optim.Adam(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
        )
        logger.log(
            f"Optimizer: Adam (lr={float(args.lr)} betas=({float(args.beta1)},{float(args.beta2)}) "
            f"eps={float(args.optimizer_eps)} grad_clip={float(args.grad_clip)} weight_decay=0)"
        )

    fsdp_enabled = (FSDP is not None) and isinstance(train_model, FSDP)
    iterator = sft_backend.ShardedIterator(data, rank=rank, world_size=world_size, seed=int(args.seed))

    tok, encode, _decode, infer_engine, _ = eval_base._build_engine(
        model_path=args.model,
        device=device,
        ctx_len=int(args.ctx_len),
        tokenizer_path=args.tokenizer,
        use_rapid_sampling=bool(args.use_rapid_sampling),
    )

    logger.log(
        f"MaxRL train begin: steps={args.max_steps} global_batch={args.global_batch_size} "
        f"local_update_group_target={local_update_group_target} group_size={cfg.group_size} "
        f"rollout_batch_size={cfg.rollout_batch_size} dynamic_prompt_batch={cfg.dynamic_prompt_batch} "
        f"max_sampling_rounds={cfg.max_sampling_rounds} sync_diag_interval={cfg.sync_diag_interval} "
        f"sync_diag_sample_values={cfg.sync_diag_sample_values} eval_interval={cfg.eval_interval} "
        f"eval_batch_size={cfg.eval_batch_size} optimizer={cfg.optimizer} lr={args.lr:.2e} "
        f"adamw_fallback_lr={cfg.adamw_fallback_lr:.2e} beta1={cfg.beta1} beta2={cfg.beta2} "
        f"optimizer_eps={cfg.optimizer_eps} muon_eps={cfg.muon_eps} grad_clip={cfg.grad_clip}"
    )
    logger.log(f"CONFIG full={json.dumps(asdict(cfg), sort_keys=True)}")
    logger.log(
        "Rollout path: eval_math192_grpo.py::_build_engine + FP16BatchInference.generate_group_parallel "
        "(single-turn, no tool use, no multi-turn)."
    )
    logger.log(
        "MaxRL path: hand-copied compute_maxrl_outcome_advantage from "
        "maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py."
    )
    logger.log(
        "WARNING: FSDP train->rollout sync is copied from grpo_math_local.py as requested, "
        "but this path may contain a severe bug and is not trusted yet."
    )
    logger.log(
        "WARNING: sync_infer_offload_cpu is enabled to match grpo_math_local.py default, "
        "but this CPU-offload sync path is UNVERIFIED in this script and may contain serious hazards."
    )
    logger.log(
        "Precision path: train forward uses torch.autocast(cuda, bfloat16); "
        "cross-entropy is computed in float32; optimizer updates remain float32-compatible. "
        "FSDP mixed_precision stays disabled by default to match sft_tool_use_fsdp.py "
        "unless RWKV_FSDP_ENABLE_MP=1 is set."
    )
    logger.log(f"Allocator path: PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    p0 = next(train_model.parameters())
    logger.log(f"model dtype={p0.dtype}, device={p0.device}")
    train_model.train()

    try:
        opt.zero_grad(set_to_none=True)
        if int(cfg.eval_interval) > 0:
            _sync_infer_weights(
                step=0,
                train_model=train_model,
                base_model=train_model_raw,
                infer_engine=infer_engine,
                fsdp_enabled=fsdp_enabled,
                world_size=world_size,
                train_gpu=train_gpu,
                logger=logger,
                is_main=is_main,
                sync_diag_interval=int(cfg.sync_diag_interval),
                sync_diag_sample_values=int(cfg.sync_diag_sample_values),
                sync_infer_offload_cpu=bool(cfg.sync_infer_offload_cpu),
            )
            _evaluate_math(
                step=0,
                eval_data=eval_data,
                infer_engine=infer_engine,
                tokenizer=tok,
                cfg=cfg,
                device=device,
                rank=rank,
                world_size=world_size,
                train_gpu=train_gpu,
                logger=logger,
                is_main=is_main,
            )
            train_model.train()

        for step in range(1, int(args.max_steps) + 1):
            step_start = time.time()
            _sync_infer_weights(
                step=step,
                train_model=train_model,
                base_model=train_model_raw,
                infer_engine=infer_engine,
                fsdp_enabled=fsdp_enabled,
                world_size=world_size,
                train_gpu=train_gpu,
                logger=logger,
                is_main=is_main,
                sync_diag_interval=int(cfg.sync_diag_interval),
                sync_diag_sample_values=int(cfg.sync_diag_sample_values),
                sync_infer_offload_cpu=bool(cfg.sync_infer_offload_cpu),
            )

            selected_groups_global = None
            prompts_per_batch = max(1, int(cfg.rollout_batch_size) // max(1, int(cfg.group_size)))
            local_round_prompt_n, _, _ = _split_global_count(int(cfg.dynamic_prompt_batch), world_size, rank)
            accumulated_valid_groups = [] if is_main else None

            for rollout_round in range(1, int(cfg.max_sampling_rounds) + 1):
                if bool(args.overfit_one) and overfit_record is not None:
                    prompt_records = [dict(overfit_record) for _ in range(local_round_prompt_n)]
                else:
                    prompt_records = [iterator.next() for _ in range(local_round_prompt_n)]
                round_prompt_groups: List[Dict[str, Any]] = []
                for rec in prompt_records:
                    round_prompt_groups.append(
                        {
                            "problem": rec["problem"],
                            "answer": rec["answer"],
                            "prompt_ids": _build_prompt_ids(rec, tok, cfg.ctx_len, cfg.max_new_tokens),
                            "samples": [],
                        }
                    )

                for start in range(0, len(round_prompt_groups), prompts_per_batch):
                    chunk = round_prompt_groups[start : start + prompts_per_batch]
                    batch_prompts = [g["prompt_ids"] for g in chunk]
                    batch_seed = int(
                        cfg.seed
                        + step * 1009
                        + rollout_round * 10007
                        + start * 1009
                        + rank * 1000003
                    )

                    comp_tokens, old_logps, comp_texts, truncated = infer_engine.generate_group_parallel(
                        prompt_tokens_list=batch_prompts,
                        group_size=int(cfg.group_size),
                        max_new_tokens=int(cfg.max_new_tokens),
                        temperature=float(cfg.temperature),
                        top_p=float(cfg.top_p),
                        top_k=int(cfg.top_k),
                        stop_on_think_close=False,
                        stop_on_user=True,
                        stop_on_boxed=False,
                        stop_check_every=8,
                        stop_check_window=96,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        penalty_decay=0.0,
                        use_rollout_cache=False,
                        rng_seed=batch_seed,
                    )

                    for i, group in enumerate(chunk):
                        gt = group["answer"]
                        for gi in range(int(cfg.group_size)):
                            idx = i * int(cfg.group_size) + gi
                            judge = _judge_gsm8k(comp_texts[idx], gt, bool(truncated[idx]))
                            reward = 1.0 if bool(judge.get("ok", False)) else 0.0
                            group["samples"].append(
                                {
                                    "comp_tokens": [int(x) for x in comp_tokens[idx]],
                                    "old_logps": [float(x) for x in old_logps[idx]],
                                    "completion": comp_texts[idx],
                                    "truncated": bool(truncated[idx]),
                                    "judge": judge,
                                    "reward": float(reward),
                                    "adv": 0.0,
                                }
                            )

                local_valid_groups = [group for group in round_prompt_groups if not _group_is_all_zero_or_all_one(group)]
                local_all0_or_all1 = int(len(round_prompt_groups) - len(local_valid_groups))
                local_sample_n, local_correct_n, local_all0_n, local_all1_n, local_mixed_n = _group_reward_counts(
                    round_prompt_groups
                )

                if world_size > 1 and _dist_is_initialized():
                    gathered_valid_groups: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_valid_groups, local_valid_groups)
                else:
                    gathered_valid_groups = [local_valid_groups]

                status_obj = None
                if is_main:
                    for gathered_part in gathered_valid_groups:
                        if isinstance(gathered_part, list) and gathered_part:
                            accumulated_valid_groups.extend(gathered_part)
                    if bool(args.overfit_one) and overfit_record is None and accumulated_valid_groups:
                        first_valid = accumulated_valid_groups[0]
                        overfit_record = {
                            "problem": str(first_valid.get("problem", "")),
                            "answer": str(first_valid.get("answer", "")),
                        }
                        logger.log(
                            f"[OVERFIT] selected problem={overfit_record['problem'][:120]} "
                            f"answer={overfit_record['answer']}"
                        )
                        accumulated_valid_groups = [
                            copy.deepcopy(first_valid) for _ in range(int(cfg.global_batch_size))
                        ]
                    valid_total = len(accumulated_valid_groups)
                    sampled_total = int(rollout_round * cfg.dynamic_prompt_batch)
                    logger.log(
                        f"[sampling step {step}] round={rollout_round}/{cfg.max_sampling_rounds} "
                        f"sampled_global={sampled_total} valid_global={valid_total}/{cfg.global_batch_size} "
                        f"round_local_acc(rank0)={local_correct_n}/{max(1, local_sample_n)}="
                        f"{float(local_correct_n) / max(1.0, float(local_sample_n)):.4f} "
                        f"round_local_groups(rank0)=mixed:{local_mixed_n} all0:{local_all0_n} all1:{local_all1_n} "
                        f"round_local_valid(rank0)={len(local_valid_groups)} round_local_all0or1(rank0)={local_all0_or_all1}"
                    )
                    if valid_total >= int(cfg.global_batch_size):
                        selected_groups_global = accumulated_valid_groups[: int(cfg.global_batch_size)]
                        status_obj = {
                            "done": True,
                            "ok": True,
                            "message": "",
                            "selected_groups": selected_groups_global,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }
                    elif rollout_round >= int(cfg.max_sampling_rounds):
                        status_obj = {
                            "done": True,
                            "ok": False,
                            "message": (
                                f"Dynamic sampling exhausted at step={step}: "
                                f"valid_groups={valid_total} < target={cfg.global_batch_size} "
                                f"after max_rounds={cfg.max_sampling_rounds}. "
                                "Treat as training collapse and terminate."
                            ),
                            "selected_groups": None,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }
                    else:
                        status_obj = {
                            "done": False,
                            "ok": True,
                            "message": "",
                            "selected_groups": None,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }

                if world_size > 1 and _dist_is_initialized():
                    status_box = [status_obj if is_main else None]
                    dist.broadcast_object_list(status_box, src=0)
                    status_obj = status_box[0]
                    overfit_box = [overfit_record if is_main else None]
                    dist.broadcast_object_list(overfit_box, src=0)
                    overfit_record = overfit_box[0]

                if not bool(status_obj["ok"]):
                    raise RuntimeError(str(status_obj["message"]))
                if bool(status_obj["done"]):
                    selected_groups_global = status_obj["selected_groups"]
                    break

            if selected_groups_global is None:
                raise RuntimeError(f"Dynamic sampling failed to produce selected_groups_global at step={step}.")

            _, shard_start, shard_end = _split_global_count(int(cfg.global_batch_size), world_size, rank)
            prompt_groups = selected_groups_global[shard_start:shard_end]
            if len(prompt_groups) != int(local_update_group_target):
                raise RuntimeError(
                    f"Rank {rank} received {len(prompt_groups)} selected groups, "
                    f"expected {local_update_group_target}."
                )

            flat_rewards = []
            flat_uid = []
            flat_refs = []
            for pi, group in enumerate(prompt_groups):
                for sample in group["samples"]:
                    flat_rewards.append([float(sample["reward"])])
                    flat_uid.append(pi)
                    flat_refs.append(sample)

            if not flat_refs:
                raise RuntimeError(f"No rollout samples collected at step={step}.")

            reward_tensor = torch.tensor(flat_rewards, dtype=torch.float32, device=device)
            reward_mask = torch.ones_like(reward_tensor, dtype=torch.float32, device=device)
            adv_tensor, _ = compute_maxrl_outcome_advantage(
                token_level_rewards=reward_tensor,
                response_mask=reward_mask,
                index=np.asarray(flat_uid),
            )
            for i, sample in enumerate(flat_refs):
                sample["adv"] = float(adv_tensor[i, 0].detach().item())

            local_target_tokens = 0
            sample_cnt_local = 0
            sample_correct_local = 0
            prompt_pass_local = 0
            reward_sum_local = 0.0
            adv_abs_sum_local = 0.0
            traj_cnt_local = 0
            for group in prompt_groups:
                group_any_ok = False
                for sample in group["samples"]:
                    keep = min(len(sample["comp_tokens"]), len(sample["old_logps"]))
                    if keep <= 0:
                        continue
                    local_target_tokens += int(keep)
                    sample_cnt_local += 1
                    ok = bool(sample["judge"].get("ok", False))
                    sample_correct_local += int(ok)
                    group_any_ok = group_any_ok or ok
                    reward_sum_local += float(sample["reward"])
                    adv_abs_sum_local += abs(float(sample["adv"]))
                    traj_cnt_local += 1
                prompt_pass_local += int(group_any_ok)

            global_target_tokens = sft_backend._all_reduce_scalar(
                local_target_tokens,
                device=device,
                op=dist.ReduceOp.SUM,
            )
            if global_target_tokens <= 0:
                raise RuntimeError(f"Resolved global_target_tokens={global_target_tokens} at step={step}.")

            opt.zero_grad(set_to_none=True)
            step_policy_local = 0.0
            delta_tok_sum_local = 0.0
            delta_tok_abs_sum_local = 0.0
            delta_seq_sum_local = 0.0
            delta_seq_abs_sum_local = 0.0
            delta_tok_cnt_local = 0
            delta_seq_cnt_local = 0
            ratio_sum_local = 0.0
            ratio_max_local = 0.0
            clip_cnt_local = 0.0

            for group in prompt_groups:
                valid_samples = []
                for sample in group["samples"]:
                    keep = min(len(sample["comp_tokens"]), len(sample["old_logps"]))
                    if keep <= 0:
                        continue
                    valid_samples.append(
                        {
                            "full_tokens": group["prompt_ids"] + sample["comp_tokens"][:keep],
                            "prompt_len": len(group["prompt_ids"]),
                            "old_logps": sample["old_logps"][:keep],
                            "adv": float(sample["adv"]),
                        }
                    )
                if not valid_samples:
                    continue

                for sample in valid_samples:
                    padded, lens = _pad_batch_local([sample["full_tokens"]], device=device, pad_id=0)
                    inp = padded[:, :-1].contiguous()
                    tgt = padded[:, 1:].contiguous()

                    with contextlib.nullcontext():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = train_model(inp)
                        if torch.is_tensor(logits) and logits.dim() == 2:
                            logits = logits.unsqueeze(0)
                        picked = -F.cross_entropy(
                            logits.float().reshape(-1, logits.size(-1)),
                            tgt.reshape(-1),
                            reduction="none",
                        ).reshape_as(tgt)

                        prompt_len = int(sample["prompt_len"])
                        full_len = int(lens[0])
                        comp_len = int(len(sample["old_logps"]))
                        start = max(0, prompt_len - 1)
                        end = max(start, min(full_len - 1, start + comp_len))
                        if end <= start:
                            continue

                        new_lp = picked[0, start:end].float()
                        old_lp = torch.tensor(sample["old_logps"], dtype=torch.float32, device=device)
                        if old_lp.numel() != new_lp.numel():
                            keep = min(old_lp.numel(), new_lp.numel())
                            if keep <= 0:
                                continue
                            new_lp = new_lp[:keep]
                            old_lp = old_lp[:keep]

                        adv = torch.full_like(new_lp, float(sample["adv"]))
                        lp_delta = new_lp - old_lp
                        ratio = torch.exp(lp_delta.clamp(min=-20.0, max=20.0))
                        obj = _ppo_clipped_objective(ratio, adv, clip_range=float(cfg.clip_range))
                        policy_sum = -obj.sum()
                        loss = policy_sum / float(global_target_tokens)
                        loss.backward()
                        step_policy_local += float(policy_sum.detach().item())
                        with torch.no_grad():
                            delta_f = lp_delta.detach().float()
                            ratio_f = ratio.detach().float()
                            tok_n = int(delta_f.numel())
                            if tok_n > 0:
                                delta_tok_sum_local += float(delta_f.sum().item())
                                delta_tok_abs_sum_local += float(delta_f.abs().sum().item())
                                seq_delta = float(delta_f.sum().item())
                                delta_seq_sum_local += seq_delta
                                delta_seq_abs_sum_local += abs(seq_delta)
                                delta_tok_cnt_local += tok_n
                                delta_seq_cnt_local += 1
                                ratio_sum_local += float(ratio_f.sum().item())
                                ratio_max_local = max(ratio_max_local, float(ratio_f.max().item()))
                                clip_cnt_local += float(
                                    (
                                        (ratio_f > (1.0 + float(cfg.clip_range)))
                                        | (ratio_f < (1.0 - float(cfg.clip_range)))
                                    )
                                    .float()
                                    .sum()
                                    .item()
                                )

            raw_grad_sq_local = 0.0
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        raw_grad_sq_local += float((g.norm(2) ** 2).item())
            raw_grad_sq = sft_backend._all_reduce_scalar(raw_grad_sq_local, device=device, op=dist.ReduceOp.SUM)
            raw_grad = math.sqrt(max(0.0, raw_grad_sq))

            if float(args.grad_clip) > 0:
                if fsdp_enabled and hasattr(train_model, "clip_grad_norm_"):
                    train_model.clip_grad_norm_(float(args.grad_clip))
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))

            grad_sq_local = 0.0
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        grad_sq_local += float((g.norm(2) ** 2).item())
            grad_sq = sft_backend._all_reduce_scalar(grad_sq_local, device=device, op=dist.ReduceOp.SUM)
            grad_norm = math.sqrt(max(0.0, grad_sq))

            opt.step()
            opt.zero_grad(set_to_none=True)

            was_training = bool(train_model.training)
            train_model.eval()
            post_stats_local = _logprob_delta_stats_local(
                train_model=train_model,
                prompt_groups=prompt_groups,
                device=device,
                clip_range=float(cfg.clip_range),
            )
            if was_training:
                train_model.train()

            step_policy = sft_backend._all_reduce_scalar(step_policy_local, device=device, op=dist.ReduceOp.SUM)
            sample_cnt = sft_backend._all_reduce_scalar(sample_cnt_local, device=device, op=dist.ReduceOp.SUM)
            sample_correct = sft_backend._all_reduce_scalar(sample_correct_local, device=device, op=dist.ReduceOp.SUM)
            prompt_pass = sft_backend._all_reduce_scalar(prompt_pass_local, device=device, op=dist.ReduceOp.SUM)
            reward_sum = sft_backend._all_reduce_scalar(reward_sum_local, device=device, op=dist.ReduceOp.SUM)
            adv_abs_sum = sft_backend._all_reduce_scalar(adv_abs_sum_local, device=device, op=dist.ReduceOp.SUM)
            traj_cnt = sft_backend._all_reduce_scalar(traj_cnt_local, device=device, op=dist.ReduceOp.SUM)
            delta_tok_sum = sft_backend._all_reduce_scalar(delta_tok_sum_local, device=device, op=dist.ReduceOp.SUM)
            delta_tok_abs_sum = sft_backend._all_reduce_scalar(delta_tok_abs_sum_local, device=device, op=dist.ReduceOp.SUM)
            delta_seq_sum = sft_backend._all_reduce_scalar(delta_seq_sum_local, device=device, op=dist.ReduceOp.SUM)
            delta_seq_abs_sum = sft_backend._all_reduce_scalar(delta_seq_abs_sum_local, device=device, op=dist.ReduceOp.SUM)
            delta_tok_cnt = sft_backend._all_reduce_scalar(delta_tok_cnt_local, device=device, op=dist.ReduceOp.SUM)
            delta_seq_cnt = sft_backend._all_reduce_scalar(delta_seq_cnt_local, device=device, op=dist.ReduceOp.SUM)
            ratio_sum = sft_backend._all_reduce_scalar(ratio_sum_local, device=device, op=dist.ReduceOp.SUM)
            ratio_max = sft_backend._all_reduce_scalar(ratio_max_local, device=device, op=dist.ReduceOp.MAX)
            clip_cnt = sft_backend._all_reduce_scalar(clip_cnt_local, device=device, op=dist.ReduceOp.SUM)
            post_delta_tok_sum = sft_backend._all_reduce_scalar(
                post_stats_local["delta_tok_sum"], device=device, op=dist.ReduceOp.SUM
            )
            post_delta_tok_abs_sum = sft_backend._all_reduce_scalar(
                post_stats_local["delta_tok_abs_sum"], device=device, op=dist.ReduceOp.SUM
            )
            post_delta_seq_sum = sft_backend._all_reduce_scalar(
                post_stats_local["delta_seq_sum"], device=device, op=dist.ReduceOp.SUM
            )
            post_delta_seq_abs_sum = sft_backend._all_reduce_scalar(
                post_stats_local["delta_seq_abs_sum"], device=device, op=dist.ReduceOp.SUM
            )
            post_delta_tok_cnt = sft_backend._all_reduce_scalar(
                post_stats_local["delta_tok_cnt"], device=device, op=dist.ReduceOp.SUM
            )
            post_delta_seq_cnt = sft_backend._all_reduce_scalar(
                post_stats_local["delta_seq_cnt"], device=device, op=dist.ReduceOp.SUM
            )
            post_ratio_sum = sft_backend._all_reduce_scalar(
                post_stats_local["ratio_sum"], device=device, op=dist.ReduceOp.SUM
            )
            post_ratio_max = sft_backend._all_reduce_scalar(
                post_stats_local["ratio_max"], device=device, op=dist.ReduceOp.MAX
            )
            post_clip_cnt = sft_backend._all_reduce_scalar(
                post_stats_local["clip_cnt"], device=device, op=dist.ReduceOp.SUM
            )
            avg_loss = step_policy / max(1.0, float(global_target_tokens))
            tok_denom = max(1.0, float(delta_tok_cnt))
            seq_denom = max(1.0, float(delta_seq_cnt))
            post_tok_denom = max(1.0, float(post_delta_tok_cnt))
            post_seq_denom = max(1.0, float(post_delta_seq_cnt))
            dt = time.time() - step_start

            if is_main and (step == 1 or step == int(args.max_steps) or step % max(1, int(args.log_interval)) == 0):
                logger.log(
                    f"[train step {step}/{int(args.max_steps)}] "
                    f"loss={avg_loss:.6f} resp_tok={int(global_target_tokens)} "
                    f"selected_sample_acc={float(sample_correct) / max(1.0, float(sample_cnt)):.4f} "
                    f"prompt_pass={float(prompt_pass) / max(1.0, float(cfg.global_batch_size)):.4f} "
                    f"reward_mean={float(reward_sum) / max(1.0, float(sample_cnt)):.4f} "
                    f"adv_abs_mean={float(adv_abs_sum) / max(1.0, float(traj_cnt)):.4f} "
                    f"pre_lp_delta_tok_mean={float(delta_tok_sum) / tok_denom:.6f} "
                    f"pre_lp_delta_tok_abs={float(delta_tok_abs_sum) / tok_denom:.6f} "
                    f"pre_lp_delta_seq_mean={float(delta_seq_sum) / seq_denom:.6f} "
                    f"pre_lp_delta_seq_abs={float(delta_seq_abs_sum) / seq_denom:.6f} "
                    f"pre_ratio_mean={float(ratio_sum) / tok_denom:.6f} "
                    f"pre_ratio_max={float(ratio_max):.6f} "
                    f"pre_clipfrac={float(clip_cnt) / tok_denom:.6f} "
                    f"post_lp_delta_tok_mean={float(post_delta_tok_sum) / post_tok_denom:.6f} "
                    f"post_lp_delta_tok_abs={float(post_delta_tok_abs_sum) / post_tok_denom:.6f} "
                    f"post_lp_delta_seq_mean={float(post_delta_seq_sum) / post_seq_denom:.6f} "
                    f"post_lp_delta_seq_abs={float(post_delta_seq_abs_sum) / post_seq_denom:.6f} "
                    f"post_ratio_mean={float(post_ratio_sum) / post_tok_denom:.6f} "
                    f"post_ratio_max={float(post_ratio_max):.6f} "
                    f"post_clipfrac={float(post_clip_cnt) / post_tok_denom:.6f} "
                    f"raw_grad={raw_grad:.6f} grad={grad_norm:.6f} "
                    f"lr={float(opt.param_groups[0]['lr']):.2e} step_time={dt:.2f}s"
                )

            if int(cfg.eval_interval) > 0 and step % int(cfg.eval_interval) == 0:
                _sync_infer_weights(
                    step=step,
                    train_model=train_model,
                    base_model=train_model_raw,
                    infer_engine=infer_engine,
                    fsdp_enabled=fsdp_enabled,
                    world_size=world_size,
                    train_gpu=train_gpu,
                    logger=logger,
                    is_main=is_main,
                    sync_diag_interval=int(cfg.sync_diag_interval),
                    sync_diag_sample_values=int(cfg.sync_diag_sample_values),
                    sync_infer_offload_cpu=bool(cfg.sync_infer_offload_cpu),
                )
                _evaluate_math(
                    step=step,
                    eval_data=eval_data,
                    infer_engine=infer_engine,
                    tokenizer=tok,
                    cfg=cfg,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    train_gpu=train_gpu,
                    logger=logger,
                    is_main=is_main,
                )

            if int(args.save_interval) > 0 and (step % int(args.save_interval) == 0 or step == int(args.max_steps)):
                model_state = sft_backend._collect_model_state(train_model, train_model_raw, fsdp_enabled=fsdp_enabled)
                if is_main and model_state is not None:
                    ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pth")
                    torch.save(
                        {
                            "time": now_str(),
                            "step": step,
                            "cfg": asdict(cfg),
                            "model_state": model_state,
                        },
                        ckpt_path,
                    )
                    latest_full_path = os.path.join(args.out_dir, "latest_full_model.pth")
                    torch.save(model_state, latest_full_path)
                    logger.log(f"saved: {ckpt_path}")
                    logger.log(f"saved: {latest_full_path}")
                if world_size > 1 and _dist_is_initialized():
                    torch.cuda.set_device(train_gpu)
                    dist.barrier(device_ids=[int(train_gpu)])

        logger.log("train end.")
    finally:
        if world_size > 1 and _dist_is_initialized():
            dist.destroy_process_group()


def _pad_batch_local(seqs: List[List[int]], device: str, pad_id: int = 0):
    lens = [len(s) for s in seqs]
    tmax = max(lens)
    bsz = len(seqs)
    x = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        if seq:
            x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return x, lens


if __name__ == "__main__":
    main()
