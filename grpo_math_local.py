#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWKV Full-Parameter GRPO training (parallel inference + local regex judge).
- Keeps original RWKV prompt style (including trailing " think")
- Supports PPO-style update and Qwen-aligned vanilla token-PG update
- Uses local-only imports for portability across machines
"""

import os
import sys

# Add paths for imports (relative for portability)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(REPO_DIR, "reference")
RAPID_SAMPLING_DIR = os.path.join(REPO_DIR, "Rapid-Sampling-main")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, REFERENCE_DIR)

import re
import json
import time
import math
import random
import pickle
import copy
import gc
import argparse
import shlex
import shutil
import threading
import functools
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:
    FSDP = None
    FullStateDictConfig = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None
    transformer_auto_wrap_policy = None


def _stochastic_roundf_to_bf16(fp32_tensor: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding from fp32 to bfloat16 (unbiased).

    bf16 drops the lower 16 mantissa bits of fp32. Adding uniform random
    noise to those 16 bits before truncation gives exact stochastic
    rounding: P(round up) = fractional_part / ULP.  This is unbiased
    (E[bf16] == fp32) and allows sub-ULP Adam updates to propagate.
    """
    bits = fp32_tensor.view(torch.int32)
    rand = torch.randint_like(bits, 0, (1 << 16))
    rounded = (bits + rand) & torch.tensor(~0xFFFF, dtype=torch.int32, device=bits.device)
    return rounded.view(torch.float32).to(torch.bfloat16)


class MemoryEfficientAdamW(torch.optim.AdamW):
    """AdamW with optimizer states offloaded to CPU pinned memory.

    Maintains fp32 master copies of parameters to avoid precision loss
    when model parameters are in bf16/fp16.  Small Adam updates (~1e-5)
    accumulate correctly in fp32, then are cast back to the model dtype.
    """

    def __init__(
        self,
        params,
        lr=1e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.pin_memory = pin_memory
        self.enabled = enabled

    @torch.no_grad()
    def step(self, closure=None):
        if not self.enabled:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            fp32_params = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    device = "cpu"
                    pin_memory = self.pin_memory
                    dtype = torch.float32

                    state["exp_avg"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, device=device, pin_memory=pin_memory, dtype=dtype
                        )
                    # fp32 master copy – needed when param is bf16/fp16
                    if p.dtype != torch.float32:
                        state["fp32_param"] = p.data.float().to(
                            device=device, copy=True
                        )
                        if pin_memory and device == "cpu":
                            state["fp32_param"] = state["fp32_param"].pin_memory()

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                fp32_params.append(state.get("fp32_param", None))

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                param_device = param.device

                exp_avg = exp_avgs[i].to(param_device, non_blocking=True)
                exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)
                step = state_steps[i]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                    max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                fp32_p = fp32_params[i]
                if fp32_p is not None:
                    # Update fp32 master copy (on GPU), then cast back to bf16
                    fp32_p_gpu = fp32_p.to(param_device, non_blocking=True)
                    if group["weight_decay"] != 0:
                        fp32_p_gpu.mul_(1 - group["lr"] * group["weight_decay"])
                    fp32_p_gpu.addcdiv_(exp_avg, denom, value=-step_size)
                    # Stochastic rounding fp32 -> bf16: unbiased, allows sub-ULP
                    # updates to propagate probabilistically each step.
                    param.data.copy_(_stochastic_roundf_to_bf16(fp32_p_gpu))
                    fp32_params[i] = None  # avoid keeping GPU ref
                    # Store updated master back to CPU
                    self.state[param]["fp32_param"].copy_(fp32_p_gpu, non_blocking=True)
                else:
                    # param is already fp32 – update in place as before
                    if group["weight_decay"] != 0:
                        param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.addcdiv_(exp_avg, denom, value=-step_size)

                exp_avgs[i].copy_(exp_avg, non_blocking=True)
                exp_avg_sqs[i].copy_(exp_avg_sq, non_blocking=True)

        return loss


# =========================================================
# Utils
# =========================================================

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _atomic_write_text(path: str, text: str):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _append_text_line(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _resolve_torch_cuda_arch_list(device: str) -> str:
    """
    Pick a safe TORCH_CUDA_ARCH_LIST for cpp_extension.load().
    Newer GPUs may report arch 10.0 while current torch parser doesn't know it.
    In that case we fallback to 9.0+PTX for forward-compatible PTX JIT.
    """
    if not str(device).startswith("cuda"):
        return ""
    try:
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        major, minor = torch.cuda.get_device_capability(idx)
        if major >= 10:
            return "9.0+PTX"
        return f"{major}.{minor}"
    except Exception:
        return ""


def _parse_cuda_index(device: str) -> Optional[int]:
    s = str(device).strip().lower()
    if not s.startswith("cuda"):
        return None
    if ":" in s:
        try:
            return int(s.split(":", 1)[1])
        except Exception:
            return None
    return 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _get_tensor_by_dotted_name(root: Any, dotted_name: str) -> Optional[torch.Tensor]:
    """
    Resolve a tensor-like attribute by dotted path from a possibly FSDP-wrapped module tree.
    Supports numeric indexing through ModuleList/list and common wrapper attrs.
    """
    if root is None or not dotted_name:
        return None
    cur = root
    parts = [p for p in str(dotted_name).split(".") if p]
    for part in parts:
        nxt = None
        # Direct attribute / index access.
        if part.isdigit():
            idx = int(part)
            try:
                nxt = cur[idx]
            except Exception:
                nxt = None
        else:
            nxt = getattr(cur, part, None)

        # Fallback through wrapper modules.
        if nxt is None:
            wrapped = getattr(cur, "module", None)
            if wrapped is not None:
                if part.isdigit():
                    idx = int(part)
                    try:
                        nxt = wrapped[idx]
                    except Exception:
                        nxt = None
                else:
                    nxt = getattr(wrapped, part, None)
        if nxt is None:
            wrapped = getattr(cur, "_fsdp_wrapped_module", None)
            if wrapped is not None:
                if part.isdigit():
                    idx = int(part)
                    try:
                        nxt = wrapped[idx]
                    except Exception:
                        nxt = None
                else:
                    nxt = getattr(wrapped, part, None)

        if nxt is None:
            return None
        cur = nxt

    if isinstance(cur, torch.nn.Parameter):
        return cur.data
    if torch.is_tensor(cur):
        return cur
    return None


def _dist_is_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _dist_rank() -> int:
    if not _dist_is_initialized():
        return 0
    try:
        return int(dist.get_rank())
    except Exception:
        return 0


def _dist_world_size() -> int:
    if not _dist_is_initialized():
        return 1
    try:
        return max(1, int(dist.get_world_size()))
    except Exception:
        return 1


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _ensure_cuda_toolkit_env():
    """
    Ensure CUDA toolkit headers are discoverable for torch cpp_extension builds.
    """
    def _extract_nvcc_bin(raw: str) -> str:
        if not raw:
            return ""
        try:
            parts = shlex.split(raw)
        except Exception:
            parts = [raw]
        if not parts:
            return ""
        # Handle wrappers like "ccache /path/to/nvcc ..."
        if os.path.basename(parts[0]) == "ccache" and len(parts) > 1:
            return parts[1]
        return parts[0]

    nvcc_candidates = []
    for key in ("CUDACXX", "NVCC"):
        val = _extract_nvcc_bin(os.environ.get(key, "").strip())
        if val:
            nvcc_candidates.append(val)
    nvcc_sys = shutil.which("nvcc")
    if nvcc_sys:
        nvcc_candidates.append(nvcc_sys)

    nvcc_real = ""
    for cand in nvcc_candidates:
        if len(cand) > 2048:
            continue
        path = os.path.realpath(cand)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            nvcc_real = path
            break

    if not nvcc_real:
        raise RuntimeError(
            "Cannot find nvcc in PATH. Please load/install CUDA toolkit (devel), not driver-only runtime."
        )
    cuda_home = os.path.dirname(os.path.dirname(nvcc_real))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    include_candidates = [
        os.path.join(cuda_home, "include"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
    ]
    if conda_prefix:
        include_candidates += [
            os.path.join(conda_prefix, "include"),
            os.path.join(conda_prefix, "targets", "x86_64-linux", "include"),
        ]
    include_candidates = [p for p in include_candidates if os.path.isdir(p)]

    chosen_include = None
    for p in include_candidates:
        if os.path.exists(os.path.join(p, "cuda_runtime.h")) and os.path.exists(os.path.join(p, "cuda_bf16.h")):
            chosen_include = p
            break
    if chosen_include is None:
        raise RuntimeError(
            f"CUDA toolkit headers not found. Tried include dirs: {include_candidates}. "
            "Need both cuda_runtime.h and cuda_bf16.h from a CUDA *devel* toolkit."
        )

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDACXX"] = nvcc_real

    def _prepend_unique_env_paths(name: str, paths: List[str]):
        cur = os.environ.get(name, "")
        cur_list = [x for x in cur.split(":") if x]
        out = []
        for p in paths + cur_list:
            if not p:
                continue
            if len(p) > 2048:
                continue
            if p not in out:
                out.append(p)
        os.environ[name] = ":".join(out)

    # Help host compiler find CUDA headers/libs in stricter cluster envs.
    _prepend_unique_env_paths("CPATH", include_candidates)
    _prepend_unique_env_paths("CPLUS_INCLUDE_PATH", include_candidates)

    lib_candidates = [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
    ]
    if conda_prefix:
        lib_candidates += [
            os.path.join(conda_prefix, "lib"),
            os.path.join(conda_prefix, "targets", "x86_64-linux", "lib"),
        ]
    lib_candidates = [p for p in lib_candidates if os.path.isdir(p)]

    _prepend_unique_env_paths("LIBRARY_PATH", lib_candidates)
    _prepend_unique_env_paths("LD_LIBRARY_PATH", lib_candidates)

    # Avoid nvcc duplicate/incompatible compiler-bindir warning.
    cxx = os.environ.get("CXX")
    if cxx:
        os.environ["CUDAHOSTCXX"] = cxx

def _normalize_record(rec: Any) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        return {"problem": str(rec), "solution": ""}
    out = dict(rec)
    if not out.get("problem"):
        for key in ("question", "prompt", "input"):
            if out.get(key):
                out["problem"] = out[key]
                break
    if not out.get("solution"):
        for key in ("answer", "output", "target"):
            if out.get(key):
                out["solution"] = out[key]
                break
    out.setdefault("problem", "")
    out.setdefault("solution", "")
    return out

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(_normalize_record(json.loads(line)))
    return data

def read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def read_parquet(path: str) -> List[Dict[str, Any]]:
    """Read parquet file and convert to list of dicts with 'problem' and 'solution' keys"""
    import pandas as pd
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        # Map 'question' -> 'problem', 'answer' -> 'solution'
        data.append({
            "problem": row.get("question", row.get("problem", "")),
            "solution": row.get("answer", row.get("solution", ""))
        })
    return data

def read_json(path: str) -> List[Dict[str, Any]]:
    """Read json file (list format) and convert to list of dicts with 'problem' and 'solution' keys"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        data.append({
            "problem": item.get("question", item.get("problem", "")),
            "solution": item.get("answer", item.get("solution", ""))
        })
    return data

def load_data(path: str) -> List[Dict[str, Any]]:
    """Load data from jsonl, parquet, or json file"""
    if path.endswith(".parquet"):
        return read_parquet(path)
    elif path.endswith(".json"):
        return read_json(path)
    else:
        return read_jsonl(path)

def filter_train_data_by_problem_len(data: List[Dict[str, Any]], max_problem_len: int = 512) -> List[Dict[str, Any]]:
    """Keep only train samples whose problem text length is <= max_problem_len."""
    if max_problem_len <= 0:
        return data
    out: List[Dict[str, Any]] = []
    for rec in data:
        problem = str(rec.get("problem", "")).strip()
        if len(problem) <= int(max_problem_len):
            out.append(rec)
    return out

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Prompt (keep 'think'!)
# =========================================================

def build_prompt(problem: str) -> str:
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"请将最终答案放在\\boxed{{...}}里，并且最终只给出\\boxed{{...}}这一行，不要输出多余内容。 think\n"
        f"Assistant: <think>\n"
    )


def build_answer_target(answer: str) -> str:
    a = str(answer or "").strip()
    return f"\\boxed{{{a}}}\n"


def build_overfit_answer_target(answer: str) -> str:
    a = str(answer or "").strip()
    return f"</think>\n\\boxed{{{a}}}\n"


def build_overfit_solution_target(solution: str, answer: str) -> str:
    sol = str(solution or "").strip()
    ans = str(answer or "").strip()
    if sol:
        boxed_pat = re.compile(r"\\boxed\{.*?\}", re.DOTALL)
        matches = list(boxed_pat.finditer(sol))
        if matches:
            last = matches[-1]
            sol = (sol[:last.start()] + ans + sol[last.end():]).strip()
        if sol:
            return sol + "\n</think>\n" + f"\\boxed{{{ans}}}\n"
    return build_overfit_answer_target(ans)


def build_overfit_prompt(problem: str) -> str:
    return build_prompt(problem)


def build_overfit_solution_prompt(problem: str) -> str:
    return build_prompt(problem)


# =========================================================
# Answer extraction & judging
# =========================================================

def _strip_math_delims(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,\;\!\:]\s*", "", s)
    return s.strip()

def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    if brace_start < 0 or brace_start >= len(text) or text[brace_start] != "{":
        return None
    depth = 0
    i = brace_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i], i
        i += 1
    return None

def extract_last_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    brace = idx + len(key) - 1
    got = _find_balanced_brace(text, brace)
    if got is None:
        return None
    inner, _ = got
    inner = _strip_math_delims(inner)
    if str(inner).strip() in ('', '...', '…', '．．．'):
        return None
    return inner

def extract_final_answer(text: str) -> Optional[str]:
    a = extract_last_boxed(text)
    if a:
        return a
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return None
    last = lines[-1].replace("</think>", "").strip()
    return _strip_math_delims(last) if last else None

def boxed_complete(text: str) -> bool:
    k = text.rfind(r"\boxed{")
    if k < 0:
        return False
    i = k + len(r"\boxed{")
    start = i
    depth = 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Check if content is placeholder like "..."
                content = text[start:i].strip()
                if content in ('...', '…', '．．．', ''):
                    return False
                return True
        i += 1
    return False

def _format_reward_rwkv(pred_text: str) -> float:
    """Format reward: checks </think> close and \boxed{} structure."""
    t = (pred_text or "").strip()
    has_think_close = "</think>" in t.lower()
    has_boxed = bool(boxed_complete(t))
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    full_boxed_last_line = False
    if lines:
        last = lines[-1]
        full_boxed_last_line = last.startswith(r"\boxed{") and bool(boxed_complete(last))
    if has_think_close and full_boxed_last_line:
        return 1.0
    r = 0.0
    if has_think_close:
        r += 0.1
    if has_boxed:
        r += 0.5
    return float(r)

def _latex_to_sympyish(s: str) -> str:
    if s is None:
        return ""
    s = _strip_math_delims(s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace("^", "**")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\infty", "oo").replace("∞", "oo")
    s = s.replace("−", "-")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    while True:
        idx = s.find(r"\frac{")
        if idx < 0:
            break
        brace1 = idx + len(r"\frac")
        got1 = _find_balanced_brace(s, brace1)
        if got1 is None:
            break
        a, end1 = got1
        if end1 + 1 >= len(s) or s[end1 + 1] != "{":
            break
        got2 = _find_balanced_brace(s, end1 + 1)
        if got2 is None:
            break
        b, end2 = got2
        s = s[:idx] + f"(({_latex_to_sympyish(a)})/({_latex_to_sympyish(b)}))" + s[end2 + 1:]

    while True:
        idx = s.find(r"\sqrt{")
        if idx < 0:
            break
        brace = idx + len(r"\sqrt")
        got = _find_balanced_brace(s, brace)
        if got is None:
            break
        inner, end = got
        s = s[:idx] + f"sqrt({_latex_to_sympyish(inner)})" + s[end + 1:]

    s = s.replace("\\", "")
    return s.strip()

# =========================================================
# Local judge (verl-aligned regex extraction)
# =========================================================

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def _normalize_final_answer_verl(final_answer: str) -> str:
    """Normalization aligned with verl.utils.reward_score.math_dapo."""
    final_answer = str(final_answer or "")
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def _fix_fracs_math_reward(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    if len(substr) < 2:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
                except Exception:
                    return string
    return new_str


def _fix_a_slash_b_math_reward(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        if string != f"{a}/{b}":
            return string
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units_math_reward(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt_math_reward(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string_math_reward(string: str) -> str:
    string = str(string or "")
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units_math_reward(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.strip()
    # Remove trailing punctuation such as "12." from chain-of-thought prose.
    string = re.sub(r"[\.\。,\，;；:：!！?？]+$", "", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt_math_reward(string)
    string = string.replace(" ", "")
    string = _fix_fracs_math_reward(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b_math_reward(string)
    return string


def _is_equiv_math_reward(str1: str, str2: str) -> bool:
    def _canonicalize_inline_fraction(s: str) -> str:
        # Canonicalize simple inline fractions like "\pi/2" -> "\frac{\pi}{2}"
        # and "x/2" -> "\frac{x}{2}" inside larger expressions.
        s = str(s or "")
        # Avoid converting already braced fractions.
        return re.sub(
            r"(?<!\\frac\{)(\\?[A-Za-z]+)\s*/\s*([0-9]+)",
            r"\\frac{\1}{\2}",
            s,
        )

    def _drop_outer_parens_if_single_expr(s: str) -> str:
        s = str(s or "").strip()
        if len(s) < 2:
            return s
        if not (s.startswith("(") and s.endswith(")")):
            return s
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
        inner = s[1:-1].strip()
        # Keep tuple-like forms unchanged.
        if "," in inner:
            return s
        return inner

    try:
        a = _drop_outer_parens_if_single_expr(_strip_string_math_reward(_canonicalize_inline_fraction(str1)))
        b = _drop_outer_parens_if_single_expr(_strip_string_math_reward(_canonicalize_inline_fraction(str2)))
        return a == b
    except Exception:
        return str(str1) == str(str2)


def _extract_answer_tail_expression(tail: str) -> Optional[str]:
    """Extract a concise math expression from the text right after 'answer' marker."""
    if tail is None:
        return None
    s = str(tail).lstrip()
    if not s:
        return None

    # "answer" must be followed by ':' or a math-ish token directly.
    if s[0] in [":", "：", "="]:
        s = s[1:].lstrip()
        if not s:
            return None
    else:
        if s[0] in [".", "。", ",", "，", ";", "；", "!", "！", "?", "？"]:
            return None
        if not re.match(r"^[\-\+\(\[\{\\0-9a-zA-Z]", s):
            return None

    # Use only first line and ignore tail after think-close.
    s = s.splitlines()[0]
    s = s.split("</think>")[0].strip()
    if not s:
        return None

    # Normalize common connective prefixes after "answer".
    # Examples:
    # - "answer is 60"
    # - "answer should be 1+2i"
    # - "answer boxed 9"
    lowered = s.lower()
    connective_patterns = [
        r"^(is|are|be|equals?)\b",
        r"^(should\s+be|would\s+be|could\s+be)\b",
        r"^(is\s+equal\s+to|equal\s+to)\b",
        r"^(box|boxed)\b",
    ]
    changed = True
    while changed:
        changed = False
        for pat in connective_patterns:
            m = re.match(pat, lowered)
            if m:
                s = s[m.end() :].lstrip(" :：=,-")
                lowered = s.lower()
                changed = True
                break
    if not s:
        return None

    # "answer in box" is an instruction, not an answer.
    if re.match(
        r"^(in\s+box|in\s+a\s+box|in\s+the\s+box|in\s+boxed(\s+form)?|with(in)?\s+\\?boxed|in\s+the\s+\\?boxed\{\}\s+notation|in\s+\\?boxed\{\}\s+notation)",
        s.lower(),
    ):
        return None

    # If there's a boxed answer in this tail, prefer that boxed value.
    boxed = extract_last_boxed(s)
    if boxed is not None and str(boxed).strip():
        return str(boxed).strip()

    # Stop at sentence separators followed by natural language.
    s = re.split(r"\.\s+(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]
    s = re.split(r"[;,，；:：]\s*(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]

    # Trim trailing punctuation.
    s = re.sub(r"[\.\。,\，;；:：!！?？]+$", "", s).strip()
    if not s:
        return None
    return s


def _extract_last_valid_answer_marker(text: str) -> Optional[str]:
    """Return the last valid answer extracted from 'answer' markers."""
    if not text:
        return None
    matches = list(re.finditer(r"(?i)\banswer\b", text))
    if not matches:
        return None

    candidates = []
    for m in matches:
        tail = text[m.end() :]
        cand = _extract_answer_tail_expression(tail)
        if cand:
            candidates.append(cand)
    if not candidates:
        return None
    return candidates[-1]


def _is_numeric_only_text(s: str) -> bool:
    s = str(s or "").strip().replace(",", "")
    return re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", s) is not None


def _extract_first_number(s: str) -> Optional[float]:
    if s is None:
        return None
    text = str(s)
    m = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _judge_with_verl_regex(
    pred_full_output: str,
    gt: str,
    truncated: bool = False,
) -> Dict[str, Any]:
    """Judge by regex extraction + normalization, aligned to verl math_dapo.

    Extraction order:
    1) last `\\boxed{...}` (as requested)
    2) fallback to `Answer: ...` regex (last match)
    """
    solution_str = str(pred_full_output or "")[-1200:]
    gt = str(gt or "")

    boxed = extract_last_boxed(solution_str)
    if boxed is not None and str(boxed).strip():
        extracted_answer = boxed
        extract_source = "boxed"
    else:
        marker_ans = _extract_last_valid_answer_marker(solution_str)
        if marker_ans is not None:
            extracted_answer = marker_ans  # use last valid answer when multiple answers exist
            extract_source = "answer_regex"
        else:
            extracted_answer = "[INVALID]"
            extract_source = "invalid"
    pred_norm = _normalize_final_answer_verl(extracted_answer)
    gt_norm = _normalize_final_answer_verl(gt)
    ok = (
        (pred_norm == gt_norm)
        or _is_equiv_math_reward(pred_norm, gt_norm)
        or _is_equiv_math_reward(extracted_answer, gt)
    )

    # If GT is numeric-only, compare numeric value only and ignore predicted units/suffix.
    if (not ok) and _is_numeric_only_text(gt_norm):
        gt_num = _extract_first_number(gt_norm)
        pred_num = _extract_first_number(extracted_answer)
        if gt_num is not None and pred_num is not None:
            ok = abs(gt_num - pred_num) <= 1e-9

    truncated_forced_zero = bool(truncated)
    if truncated_forced_zero:
        ok = False

    return {
        "ok": bool(ok),
        "raw": extracted_answer,
        "extract_source": extract_source,
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
        "error": None,
        "truncated_forced_zero": truncated_forced_zero,
    }


def _write_jsonl_atomic(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _make_request_id(prefix: str) -> str:
    return f"{prefix}_{now_str()}_{random.randint(1000, 9999)}"


class FileJudgeClient:
    def __init__(self, judge_dir: str, timeout_s: float, poll_interval_s: float, log_fn=None):
        self.judge_dir = judge_dir
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.log_fn = log_fn
        # Legacy fields retained for compatibility; local judge does not use filesystem queue.
        self.req_dir = os.path.join(judge_dir, "requests")
        self.resp_dir = os.path.join(judge_dir, "responses")

    def _log(self, msg: str):
        if self.log_fn is not None:
            self.log_fn(msg)

    def judge(self, items: List[Dict[str, Any]], tag: str) -> Dict[str, Dict[str, Any]]:
        if not items:
            return {}
        results = {}
        for item in items:
            item_id = item.get("item_id")
            if item_id is None:
                continue
            rec = _judge_with_verl_regex(
                item.get("pred", ""),
                item.get("gt", ""),
                truncated=bool(item.get("truncated", False)),
            )
            rec["item_id"] = item_id
            results[item_id] = rec
        self._log(f"[JUDGE] local regex judge tag={tag} items={len(results)}")
        return results


def run_judge_service(
    judge_dir: str,
    max_workers: int = 16,
    loop_sleep_s: float = 1.0,
    once: bool = False,
):
    _ = (judge_dir, max_workers, loop_sleep_s, once)
    print("[JUDGE] no-op: local regex judge is used in-process; separate judge service is disabled.", flush=True)


# =========================================================
# Config
# =========================================================


@dataclass
class GRPOConfig:
    # Sampling / batch
    batch_prompts: int = 128
    group_size: int = 16
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    mask_token0: bool = False
    decay: float = 0.995

    # Rapid-Sampling repetition penalty
    use_rapid_sampling: bool = True
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    penalty_decay: float = 0.0

    # Stop checks
    stop_on_think_close: bool = False  # Deprecated no-op: think-close no longer stops generation
    stop_on_user: bool = False
    stop_on_boxed: bool = False
    stop_check_every: int = 8
    stop_check_window: int = 96

    # GRPO rollout / PPO
    max_rollout_rounds: int = 1
    ppo_epochs: int = 1
    lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    entropy_coef: float = 0.0
    format_reward_coef: float = 0.0
    online_correct_only_ce: bool = True
    clip_range: float = 0.2
    grad_clip: float = 100.0
    optimizer_eps: float = 1e-8

    # Training objective alignment
    use_ppo_loss: bool = True
    advantage_eps: float = 1e-6
    memory_efficient_adamw: bool = True

    ppo_max_token_len_per_gpu: int = 1024

    # Diagnostics
    diag_inner_update: bool = False
    diag_zero_padding: bool = False
    diag_compare_global_grad: bool = False
    diag_compare_step: int = 1
    diag_compare_epoch: int = 1
    diag_compare_param_count: int = 6

    # Rollout policy cache / sync
    rollout_update_interval: int = 1
    rollout_use_cache: bool = True
    rollout_ema_decay: float = 0.0
    sync_infer_interval: int = 1
    sync_infer_offload_cpu: bool = True

    # Logging / save
    log_interval: int = 1
    save_interval: int = 20
    infer_check_interval: int = 200

    # Eval
    eval_interval: int = 5
    eval_n: int = 192
    eval_temperature: float = 1.0
    eval_top_p: float = 1.0
    eval_top_k: int = 0
    eval_max_new_tokens: int = 1024
    eval_presence_penalty: float = 0.0
    eval_frequency_penalty: float = 0.0
    eval_penalty_decay: float = 0.0
    eval_before_train: bool = False
    overfit_test: bool = False
    overfit_batch_n: int = 16
    overfit_max_rounds: int = 20
    overfit_probe_only: bool = False

    # faulthandler
    enable_faulthandler: bool = False
    hang_dump_s: float = 0.0


# =========================================================
# Model helpers
# =========================================================

HEAD_SIZE = 64

def normalize_model_arg(model_arg: str) -> Tuple[str, str]:
    model_arg = model_arg.strip()
    if model_arg.endswith(".pth"):
        base = model_arg[:-4]
        pth = model_arg
    else:
        base = model_arg
        pth = model_arg + ".pth"
    if not os.path.isfile(pth) and os.path.isfile(base):
        pth = model_arg
        if pth.endswith(".pth"):
            base = pth[:-4]
    return base, pth

def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int, grad_cp: int = 0):
    from types import SimpleNamespace
    from rwkv7_trainable import RWKV7

    sd = _torch_load_weights(pth_path)

    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim_ffn = sd.get("blocks.0.ffn.key.weight", torch.zeros(n_embd * 4, n_embd)).shape[0]

    args = SimpleNamespace(
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_layer=n_layer,
        dim_att=n_embd,
        dim_ffn=dim_ffn,
        head_size_a=HEAD_SIZE,
        head_size_divisor=8,
        ctx_len=ctx_len,
        chunk_ctx=ctx_len,
        grad_cp=grad_cp,
        train_type="fullstate",
        peft="none",
        my_testing="x070",
    )

    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(device)  # Keep fp32 for optimizer precision (bf16 ULP too large for lr=1e-5)
    return model, args

def load_infer_model_fp16(base_name_no_pth: str, device: str = "cuda"):
    """Load optimized FP16 inference model with v2 kernel"""
    import types
    import importlib

    # Keep torch extension cache local to workspace for stable, fast reuse.
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(REPO_DIR, ".torch_extensions")

    t_import = time.time()
    print(f"[infer] importing rwkv7_fp16 ... ext_dir={os.environ.get('TORCH_EXTENSIONS_DIR')}", flush=True)
    rwkv7_fp16_mod = importlib.import_module("rwkv7_fp16")
    RWKV_x070 = rwkv7_fp16_mod.RWKV_x070
    print(f"[infer] import rwkv7_fp16 done in {time.time() - t_import:.1f}s", flush=True)

    infer_idx = _parse_cuda_index(device)
    if infer_idx is not None:
        torch.cuda.set_device(infer_idx)
    print(f"[infer] loading fp16 rollout model on {device} ...", flush=True)

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    t_model = time.time()
    model = RWKV_x070(args)
    print(f"[infer] rollout model ready on {device} in {time.time() - t_model:.1f}s", flush=True)

    return model, args

def enable_full_finetune(model: torch.nn.Module) -> int:
    cnt = 0
    for _, p in model.named_parameters():
        p.requires_grad = True
        cnt += p.numel()
    return cnt

def load_model_init(model: torch.nn.Module, path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    sd = _torch_load_weights(path)
    if "model_state" in sd and isinstance(sd["model_state"], dict):
        sd = sd["model_state"]
    elif "full_state" in sd and isinstance(sd["full_state"], dict):
        sd = sd["full_state"]
    if not isinstance(sd, dict):
        return False

    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[model_init] loaded={len(sd)} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return True


# =========================================================
# FP16 batched inference (parallel group sampling)
# =========================================================

class FP16BatchInference:
    def __init__(self, infer_model, train_model, encode_fn, decode_fn, device: str, cfg: GRPOConfig):
        self.infer_model = infer_model
        self.train_model = train_model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.device_index = _parse_cuda_index(device)
        self.cfg = cfg
        self.vocab_size = infer_model.args.vocab_size
        self.rollout_time_state = None
        self.rollout_time_state_ema = None
        self._latest_train_sd = None
        self._bad_token_warn_count = 0
        self._bad_time_state_warn_count = 0
        self._bad_vec_state_warn_count = 0

        # Load Rapid-Sampling kernel if enabled
        self.sample_kernel = None
        if cfg.use_rapid_sampling:
            from torch.utils.cpp_extension import load
            print("Loading Rapid-Sampling kernel...")
            self.sample_kernel = load(
                name="sample",
                sources=[f"{RAPID_SAMPLING_DIR}/sampling.cpp", f"{RAPID_SAMPLING_DIR}/sampling.cu"],
                extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
                verbose=False,
            )
            print("Rapid-Sampling kernel loaded.")
        self._last_sync_step = -1

    def _layer_time_state_from_sd(self, train_sd: Dict[str, torch.Tensor], layer_idx: int):
        ts = train_sd.get(f"blocks.{layer_idx}.att.time_state")
        att_ts = train_sd.get(f"blocks.{layer_idx}.att.ts_state")
        ffn_ts = train_sd.get(f"blocks.{layer_idx}.ffn.ts_state")
        return ts, att_ts, ffn_ts

    def _coerce_time_state_shape(self, ts: Optional[torch.Tensor], layer_idx: int) -> Optional[torch.Tensor]:
        if not torch.is_tensor(ts):
            return None
        args = self.infer_model.args
        H = int(args.n_embd // args.head_size)
        N = int(args.head_size)
        x = ts.detach()
        expect_numel = H * N * N

        if x.dim() == 3 and tuple(x.shape) == (H, N, N):
            return x
        if x.dim() == 3 and x.numel() == expect_numel:
            return x.reshape(H, N, N)
        if x.dim() == 2 and x.shape == (H * N, N):
            return x.reshape(H, N, N)
        if x.dim() == 2 and x.shape == (H, N * N):
            return x.reshape(H, N, N)
        if x.numel() == expect_numel:
            return x.reshape(H, N, N)

        if self._bad_time_state_warn_count < 8:
            self._bad_time_state_warn_count += 1
            print(
                f"[WARN] skip invalid time_state at layer={layer_idx}: got_shape={tuple(x.shape)} "
                f"expect=({H},{N},{N})",
                flush=True,
            )
        return None

    def _coerce_vec_state(self, x: Optional[torch.Tensor], layer_idx: int, name: str, expected_dim: int) -> Optional[torch.Tensor]:
        if not torch.is_tensor(x):
            return None
        v = x.detach().reshape(-1)
        if v.numel() == int(expected_dim):
            return v
        if self._bad_vec_state_warn_count < 8:
            self._bad_vec_state_warn_count += 1
            print(
                f"[WARN] skip invalid {name} at layer={layer_idx}: got_shape={tuple(x.shape)} "
                f"expect=({expected_dim},)",
                flush=True,
            )
        return None

    def _snapshot_time_state_from_sd(self, train_sd: Dict[str, torch.Tensor]):
        if not isinstance(train_sd, dict):
            return None
        args = self.infer_model.args
        snap = []
        for i in range(int(args.n_layer)):
            ts, att_ts, ffn_ts = self._layer_time_state_from_sd(train_sd, i)
            if ts is None:
                return None
            ts = ts.detach().clone()
            att_ts = att_ts.detach().clone() if torch.is_tensor(att_ts) else None
            ffn_ts = ffn_ts.detach().clone() if torch.is_tensor(ffn_ts) else None
            snap.append((ts, att_ts, ffn_ts))
        return snap

    def _snapshot_time_state(self):
        snap_from_sd = self._snapshot_time_state_from_sd(self._latest_train_sd)
        if snap_from_sd is not None:
            return snap_from_sd
        snap = []
        for block in self.train_model.blocks:
            ts = block.att.time_state.detach().clone()
            att_ts = block.att.ts_state.detach().clone() if hasattr(block.att, "ts_state") else None
            ffn_ts = block.ffn.ts_state.detach().clone() if hasattr(block.ffn, "ts_state") else None
            snap.append((ts, att_ts, ffn_ts))
        return snap

    @torch.no_grad()
    def sync_infer_weights(
        self,
        step: int,
        force: bool = False,
        train_sd: Optional[Dict[str, torch.Tensor]] = None,
        train_tensor_getter: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
        n_layers: Optional[int] = None,
    ):
        if (not force) and self.cfg.sync_infer_interval > 0:
            if self._last_sync_step >= 0 and (step - self._last_sync_step) < int(self.cfg.sync_infer_interval):
                return

        if train_tensor_getter is None and train_sd is None:
            train_sd = self.train_model.state_dict()

        transpose_keys = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")
        old_z = self.infer_model.z
        hit = 0

        if train_tensor_getter is not None:
            # Keep a compact time-state snapshot for rollout cache updates.
            ts_sd = {}
            nl = int(n_layers) if n_layers is not None else int(getattr(self.infer_model.args, "n_layer", 0))
            for i in range(max(0, nl)):
                for key in (
                    f"blocks.{i}.att.time_state",
                    f"blocks.{i}.att.ts_state",
                    f"blocks.{i}.ffn.ts_state",
                ):
                    t = train_tensor_getter(key)
                    if torch.is_tensor(t):
                        ts_sd[key] = t.detach().cpu().contiguous()
            self._latest_train_sd = ts_sd
        else:
            self._latest_train_sd = train_sd

        z_new = {}
        for name in list(old_z.keys()):
            src = train_tensor_getter(name) if train_tensor_getter is not None else train_sd.get(name)
            if src is None:
                z_new[name] = old_z[name]
                continue
            x = src.detach()
            if any(k in name for k in transpose_keys):
                x = x.t()
            x = x.squeeze()
            if name.endswith("att.r_k"):
                x = x.flatten()
            z_new[name] = x.to(device=self.device, dtype=torch.half).contiguous()
            hit += 1

        if "emb.weight" in z_new and "blocks.0.ln0.weight" in z_new and "blocks.0.ln0.bias" in z_new:
            z_new["emb.weight"] = F.layer_norm(
                z_new["emb.weight"],
                (self.infer_model.args.n_embd,),
                weight=z_new["blocks.0.ln0.weight"],
                bias=z_new["blocks.0.ln0.bias"],
            )
        if "blocks.0.att.a0" in z_new:
            z_new["blocks.0.att.v0"] = z_new["blocks.0.att.a0"]
        if "blocks.0.att.a1" in z_new:
            z_new["blocks.0.att.v1"] = z_new["blocks.0.att.a1"]
        if "blocks.0.att.a2" in z_new:
            z_new["blocks.0.att.v2"] = z_new["blocks.0.att.a2"]

        self.infer_model.z = z_new
        self._last_sync_step = int(step)
        if self.device_index is not None:
            torch.cuda.synchronize(self.device_index)
        print(f"[sync] infer weights synced: step={step} tensors={hit}", flush=True)

    @torch.no_grad()
    def update_rollout_time_state(self, ema_decay: float = 0.0):
        decay = float(ema_decay)
        use_ema = 0.0 < decay < 1.0
        if not use_ema:
            self.rollout_time_state = self._snapshot_time_state()
            self.rollout_time_state_ema = None
            return

        alpha = 1.0 - decay
        if self.rollout_time_state_ema is None:
            self.rollout_time_state_ema = self._snapshot_time_state()
        else:
            for i, block in enumerate(self.train_model.blocks):
                ema_ts, ema_att_ts, ema_ffn_ts = self.rollout_time_state_ema[i]
                cur_ts = block.att.time_state.detach()
                ema_ts.mul_(decay).add_(cur_ts, alpha=alpha)
                if ema_att_ts is not None:
                    cur_att = block.att.ts_state.detach()
                    ema_att_ts.mul_(decay).add_(cur_att, alpha=alpha)
                if ema_ffn_ts is not None:
                    cur_ffn = block.ffn.ts_state.detach()
                    ema_ffn_ts.mul_(decay).add_(cur_ffn, alpha=alpha)

        self.rollout_time_state = self.rollout_time_state_ema

    def init_state_with_time_state(self, B: int, time_state_snapshot=None):
        # Use the device passed to this class
        infer_device = self.device
        args = self.infer_model.args

        # Create state tensors on the correct device
        state = [None, None]
        DTYPE = torch.half  # fp16
        state[0] = torch.zeros((args.n_layer, 2, B, args.n_embd), dtype=DTYPE, requires_grad=False, device=infer_device)
        state[1] = torch.zeros((args.n_layer, B, args.n_embd // args.head_size, args.head_size, args.head_size),
                               dtype=DTYPE, requires_grad=False, device=infer_device)

        # Load trained time_state
        for i in range(int(args.n_layer)):
            ts = None
            att_ts = None
            ffn_ts = None
            if time_state_snapshot is not None and i < len(time_state_snapshot):
                ts, att_ts, ffn_ts = time_state_snapshot[i]
            elif isinstance(self._latest_train_sd, dict):
                ts, att_ts, ffn_ts = self._layer_time_state_from_sd(self._latest_train_sd, i)
            elif hasattr(self.train_model, "blocks") and i < len(self.train_model.blocks):
                block = self.train_model.blocks[i]
                ts = block.att.time_state
                att_ts = block.att.ts_state if hasattr(block.att, "ts_state") else None
                ffn_ts = block.ffn.ts_state if hasattr(block.ffn, "ts_state") else None
            if ts is None:
                continue
            ts3 = self._coerce_time_state_shape(ts, layer_idx=i)
            if ts3 is None:
                continue
            # Convert bfloat16 -> fp16 and move to infer device
            state[1][i] = ts3.unsqueeze(0).expand(B, -1, -1, -1).clone().to(device=infer_device, dtype=torch.half)
            if att_ts is not None:
                att_vec = self._coerce_vec_state(att_ts, layer_idx=i, name="att.ts_state", expected_dim=int(args.n_embd))
                if att_vec is not None:
                    att_vec = att_vec.to(device=infer_device, dtype=torch.half)
                    state[0][i, 0] = att_vec.unsqueeze(0).expand(B, -1).clone()
            if ffn_ts is not None:
                ffn_vec = self._coerce_vec_state(ffn_ts, layer_idx=i, name="ffn.ts_state", expected_dim=int(args.n_embd))
                if ffn_vec is not None:
                    ffn_vec = ffn_vec.to(device=infer_device, dtype=torch.half)
                    state[0][i, 1] = ffn_vec.unsqueeze(0).expand(B, -1).clone()
        return state

    @torch.no_grad()
    def prime_prompts(self, prompt_tokens_list: List[List[int]], time_state_snapshot=None):
        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        B = len(prompt_tokens_list)
        state = self.init_state_with_time_state(B, time_state_snapshot=time_state_snapshot)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool,
        stop_on_user: bool,
        stop_on_boxed: bool,
        stop_check_every: int,
        stop_check_window: int,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        penalty_decay: float = None,
        use_rollout_cache: bool = False,
        rng_seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:

        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        if rng_seed is not None:
            torch.cuda.manual_seed(int(rng_seed))

        # Use passed params or fall back to cfg defaults
        presence_penalty = presence_penalty if presence_penalty is not None else self.cfg.presence_penalty
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.cfg.repetition_penalty
        penalty_decay = penalty_decay if penalty_decay is not None else self.cfg.penalty_decay

        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []

        if self.vocab_size > 0:
            sanitized_prompts = []
            bad_cnt = 0
            vmax = int(self.vocab_size) - 1
            for ids in prompt_tokens_list:
                clean = []
                for t in ids:
                    ti = int(t)
                    if ti < 0 or ti > vmax:
                        bad_cnt += 1
                        ti = 0
                    clean.append(ti)
                sanitized_prompts.append(clean)
            if bad_cnt > 0 and self._bad_token_warn_count < 5:
                self._bad_token_warn_count += 1
                print(
                    f"[WARN] clamped {bad_cnt} invalid prompt token ids to 0 (vocab_size={self.vocab_size})",
                    flush=True,
                )
            prompt_tokens_list = sanitized_prompts

        time_state_snapshot = None
        if use_rollout_cache:
            if self.rollout_time_state is None:
                self.update_rollout_time_state()
            time_state_snapshot = self.rollout_time_state

        last_logits, state = self.prime_prompts(prompt_tokens_list, time_state_snapshot=time_state_snapshot)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()

        # repeat state for group
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]

        # Setup for Rapid-Sampling
        if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
            seed_now = int(rng_seed) if rng_seed is not None else int(time.time())
            rand_states = self.sample_kernel.setup_rand(seed_now, B)
            # vocab_size needs to be multiple of 4
            vocab_padded = ((self.vocab_size + 3) // 4) * 4
            penalties = torch.zeros(B, vocab_padded, device=self.device, dtype=torch.float32)

        def sample_next(logits_2d: torch.Tensor) -> torch.Tensor:
            # Use Rapid-Sampling kernel if available
            if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
                logits_float = logits_2d.float()
                # Pad logits to multiple of 4
                if logits_float.size(-1) % 4 != 0:
                    pad_size = 4 - (logits_float.size(-1) % 4)
                    logits_float = F.pad(logits_float, (0, pad_size), value=-1e30)

                return self.sample_kernel.batch_sampling_repetition_temperature_topk_topp(
                    logits_float,
                    penalties,
                    rand_states,
                    presence_penalty,
                    frequency_penalty,
                    penalty_decay,
                    temperature,
                    top_k,
                    top_p
                )

            # Fallback to original sampling
            if temperature <= 0:
                return torch.argmax(logits_2d, dim=-1)

            x = logits_2d.float() / float(temperature)
            V = x.size(-1)

            if self.cfg.mask_token0:
                x[:, 0] = -1e30

            k_cap = 0
            if top_k and top_k > 0:
                k_cap = int(min(top_k, V))
            elif top_p and 0.0 < top_p < 1.0:
                k_cap = int(min(2048, V))

            if k_cap > 0:
                topv, topi = torch.topk(x, k=k_cap, dim=-1)
                if top_p and 0.0 < top_p < 1.0:
                    probs = F.softmax(topv, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    keep = cdf <= float(top_p)
                    keep[:, 0] = True
                    topv = topv.masked_fill(~keep, -1e30)
                probs = F.softmax(topv, dim=-1)
                pick = torch.multinomial(probs, 1).squeeze(-1)
                return topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1)

            probs = F.softmax(x, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        for t in range(max_new_tokens):
            if not bool(active.any().item()):
                break

            logits = last_logits
            token_ids = sample_next(logits).long()  # Ensure int64 for gather

            logp_all = F.log_softmax(logits.float(), dim=-1)
            picked_logp = logp_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                comp_tokens[i].append(int(tok_cpu[i]))
                old_logps[i].append(float(lp_cpu[i]))

            # stop token check (0, 261, 24281)
            STOP_TOKENS = {0, 261, 24281}
            for i in range(B):
                if active[i] and int(tok_cpu[i]) in STOP_TOKENS:
                    active[i] = False

            # stop check
            if (stop_on_user or stop_on_boxed) and (t % max(1, stop_check_every) == 0):
                for i in range(B):
                    if not active[i]:
                        continue
                    w = comp_tokens[i][-stop_check_window:] if stop_check_window > 0 else comp_tokens[i]
                    s = self.decode(w)
                    if stop_on_boxed and boxed_complete(s):
                        active[i] = False
                        continue
                    if stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[i] = False
                        continue

            # Use [tok, tok, ...] format to hit optimized forward_one_batch path
            step_tokens_batch = [int(x) for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        for i in range(B):
            if bool(active[i].item()):
                truncated[i] = True

        # Clean up Rapid-Sampling tensors to free memory
        if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
            del penalties, rand_states
        if self.device_index is not None:
            torch.cuda.synchronize(self.device_index)
        torch.cuda.empty_cache()

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_tokens, old_logps, comp_text, truncated


class MultiGPUInference:
    """Dispatch rollout/eval inference across multiple infer GPUs."""

    def __init__(self, engines: List[FP16BatchInference], base_seed: int = 42):
        if not engines:
            raise RuntimeError("MultiGPUInference requires at least one FP16BatchInference engine.")
        self.engines = engines
        self.base_seed = int(base_seed)
        self._call_idx = 0
        self._call_lock = threading.Lock()

    @property
    def rollout_time_state(self):
        return self.engines[0].rollout_time_state

    @torch.no_grad()
    def should_sync(self, step: int, force: bool = False) -> bool:
        if force:
            return True
        if not self.engines:
            return False
        e0 = self.engines[0]
        interval = int(getattr(e0.cfg, "sync_infer_interval", 1))
        if interval <= 0:
            return True
        return not (e0._last_sync_step >= 0 and (int(step) - int(e0._last_sync_step)) < interval)

    @torch.no_grad()
    def sync_infer_weights(
        self,
        step: int,
        force: bool = False,
        train_sd: Optional[Dict[str, torch.Tensor]] = None,
        train_tensor_getter: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
        n_layers: Optional[int] = None,
    ):
        for engine in self.engines:
            engine.sync_infer_weights(
                step=step,
                force=force,
                train_sd=train_sd,
                train_tensor_getter=train_tensor_getter,
                n_layers=n_layers,
            )

    @torch.no_grad()
    def update_rollout_time_state(self, ema_decay: float = 0.0):
        for engine in self.engines:
            engine.update_rollout_time_state(ema_decay=ema_decay)

    def _next_seed_base(self, rng_seed: Optional[int]) -> int:
        if rng_seed is not None:
            return int(rng_seed)
        with self._call_lock:
            call_idx = self._call_idx
            self._call_idx += 1
        return int(self.base_seed + call_idx * 104729)

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool,
        stop_on_user: bool,
        stop_on_boxed: bool,
        stop_check_every: int,
        stop_check_window: int,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        penalty_decay: float = None,
        use_rollout_cache: bool = False,
        rng_seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:
        n_prompts = len(prompt_tokens_list)
        if n_prompts == 0:
            return [], [], [], []
        if len(self.engines) == 1:
            return self.engines[0].generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_on_think_close=stop_on_think_close,
                stop_on_user=stop_on_user,
                stop_on_boxed=stop_on_boxed,
                stop_check_every=stop_check_every,
                stop_check_window=stop_check_window,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                penalty_decay=penalty_decay,
                use_rollout_cache=use_rollout_cache,
                rng_seed=rng_seed,
            )

        n_engines = len(self.engines)
        group_size = max(1, int(group_size))
        base = group_size // n_engines
        rem = group_size % n_engines
        group_counts = [base + (1 if ei < rem else 0) for ei in range(n_engines)]
        active = [(ei, c) for ei, c in enumerate(group_counts) if c > 0]
        if not active:
            raise RuntimeError(f"Invalid group split: group_size={group_size}, infer_engines={n_engines}")

        offsets = [0] * n_engines
        acc = 0
        for ei, c in enumerate(group_counts):
            offsets[ei] = acc
            acc += c

        total_samples = n_prompts * group_size
        comp_tokens_out: List[Optional[List[int]]] = [None] * total_samples
        old_logps_out: List[Optional[List[float]]] = [None] * total_samples
        comp_text_out: List[Optional[str]] = [None] * total_samples
        trunc_out: List[Optional[bool]] = [None] * total_samples

        seed_base = self._next_seed_base(rng_seed)

        def _run_shard(engine: FP16BatchInference, local_group_size: int, local_seed: int):
            return engine.generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=local_group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_on_think_close=stop_on_think_close,
                stop_on_user=stop_on_user,
                stop_on_boxed=stop_on_boxed,
                stop_check_every=stop_check_every,
                stop_check_window=stop_check_window,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                penalty_decay=penalty_decay,
                use_rollout_cache=use_rollout_cache,
                rng_seed=local_seed,
            )

        for ei, local_group_size in active:
            local_seed = int(seed_base + ei * 1009)
            try:
                c_tok, o_lp, c_txt, trunc = _run_shard(self.engines[ei], int(local_group_size), local_seed)
            except Exception as e:
                raise RuntimeError(f"infer shard failed on engine#{ei}: {e}") from e

            exp_n = n_prompts * local_group_size
            if not (len(c_tok) == len(o_lp) == len(c_txt) == len(trunc) == exp_n):
                raise RuntimeError(
                    f"infer shard size mismatch on engine#{ei}: got "
                    f"{len(c_tok)}/{len(o_lp)}/{len(c_txt)}/{len(trunc)}, expected {exp_n}"
                )

            gi_start = offsets[ei]
            for pi in range(n_prompts):
                for lgi in range(local_group_size):
                    gi = gi_start + lgi
                    gidx = pi * group_size + gi
                    lidx = pi * local_group_size + lgi
                    comp_tokens_out[gidx] = c_tok[lidx]
                    old_logps_out[gidx] = o_lp[lidx]
                    comp_text_out[gidx] = c_txt[lidx]
                    trunc_out[gidx] = bool(trunc[lidx])

        if any(x is None for x in comp_tokens_out):
            raise RuntimeError("infer merge failed: incomplete comp_tokens")
        if any(x is None for x in old_logps_out):
            raise RuntimeError("infer merge failed: incomplete old_logps")
        if any(x is None for x in comp_text_out):
            raise RuntimeError("infer merge failed: incomplete comp_text")
        if any(x is None for x in trunc_out):
            raise RuntimeError("infer merge failed: incomplete trunc flags")

        return (
            [x for x in comp_tokens_out],
            [x for x in old_logps_out],
            [x for x in comp_text_out],
            [bool(x) for x in trunc_out],
        )


class TrainModelRolloutEngine:
    """Rollout with the training model itself (single-model path, no train->infer sync)."""

    def __init__(
        self,
        model: torch.nn.Module,
        decode_fn,
        device: str,
        cfg: GRPOConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.decode = decode_fn
        self.device = device
        self.device_index = _parse_cuda_index(device)
        self.cfg = cfg
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self._last_sync_step = -1
        self.rollout_micro_bsz = 4

        unwrapped = _unwrap_model(model)
        self.ctx_len = int(getattr(getattr(unwrapped, "args", None), "ctx_len", 4096))

    @property
    def rollout_time_state(self):
        return None

    @torch.no_grad()
    def should_sync(self, step: int, force: bool = False) -> bool:
        return False

    @torch.no_grad()
    def sync_infer_weights(
        self,
        step: int,
        force: bool = False,
        train_sd: Optional[Dict[str, torch.Tensor]] = None,
        train_tensor_getter: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
        n_layers: Optional[int] = None,
    ):
        self._last_sync_step = int(step)
        return

    @torch.no_grad()
    def update_rollout_time_state(self, ema_decay: float = 0.0):
        return

    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
        lens = [len(s) for s in seqs]
        T = max(1, max(lens) if lens else 1)
        B = len(seqs)
        x = torch.full((B, T), int(pad_id), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return x, lens

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool,
        stop_on_user: bool,
        stop_on_boxed: bool,
        stop_check_every: int,
        stop_check_window: int,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        penalty_decay: float = None,
        use_rollout_cache: bool = False,
        rng_seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:
        del stop_on_think_close
        del use_rollout_cache

        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        self.model.eval()

        presence_penalty = float(self.cfg.presence_penalty if presence_penalty is None else presence_penalty)
        frequency_penalty = float(self.cfg.repetition_penalty if frequency_penalty is None else frequency_penalty)
        penalty_decay = float(self.cfg.penalty_decay if penalty_decay is None else penalty_decay)
        group_size = max(1, int(group_size))

        n_prompts = len(prompt_tokens_list)
        if n_prompts == 0:
            return [], [], [], []

        full_tokens: List[List[int]] = []
        for p in prompt_tokens_list:
            base = [int(x) for x in p]
            if not base:
                base = [0]
            for _ in range(group_size):
                full_tokens.append(list(base))

        B = len(full_tokens)
        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active: List[bool] = [True for _ in range(B)]
        truncated: List[bool] = [False for _ in range(B)]
        STOP_TOKENS = {0, 261, 24281}

        apply_penalty = (presence_penalty != 0.0) or (frequency_penalty != 0.0)
        rep_counts: List[Dict[int, float]] = [{} for _ in range(B)] if apply_penalty else []

        sample_gen = None
        if rng_seed is not None:
            sample_gen = torch.Generator(device=self.device)
            sample_gen.manual_seed(int(rng_seed))

        def _sample_from_logits(logits_2d: torch.Tensor, batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
            x = logits_2d.float()
            if self.cfg.mask_token0 and x.size(-1) > 0:
                x[:, 0] = -1e30

            if apply_penalty:
                for li, gi in enumerate(batch_indices):
                    cnt = rep_counts[gi]
                    if not cnt:
                        continue
                    for tok_id, cval in cnt.items():
                        if 0 <= int(tok_id) < x.size(-1):
                            x[li, int(tok_id)] -= float(presence_penalty + frequency_penalty * cval)

            if temperature <= 0:
                tok = torch.argmax(x, dim=-1)
                logp_all = F.log_softmax(x, dim=-1)
                lp = logp_all.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                return tok.long(), lp.float()

            x = x / float(temperature)
            V = x.size(-1)
            k_cap = 0
            if top_k and int(top_k) > 0:
                k_cap = int(min(int(top_k), V))
            elif top_p and 0.0 < float(top_p) < 1.0:
                k_cap = int(min(2048, V))

            if k_cap > 0:
                topv, topi = torch.topk(x, k=k_cap, dim=-1)
                if top_p and 0.0 < float(top_p) < 1.0:
                    probs = F.softmax(topv, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    keep = cdf <= float(top_p)
                    keep[:, 0] = True
                    topv = topv.masked_fill(~keep, -1e30)
                probs = F.softmax(topv, dim=-1)
                if sample_gen is None:
                    pick = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    pick = torch.multinomial(probs, 1, generator=sample_gen).squeeze(-1)
                tok = topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
                logp_all = F.log_softmax(x, dim=-1)
                lp = logp_all.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                return tok.long(), lp.float()

            probs = F.softmax(x, dim=-1)
            if sample_gen is None:
                tok = torch.multinomial(probs, 1).squeeze(-1)
            else:
                tok = torch.multinomial(probs, 1, generator=sample_gen).squeeze(-1)
            logp_all = F.log_softmax(x, dim=-1)
            lp = logp_all.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            return tok.long(), lp.float()

        max_new_tokens = max(0, int(max_new_tokens))
        stop_check_every = max(1, int(stop_check_every))

        for t in range(max_new_tokens):
            active_idx = [i for i, flag in enumerate(active) if flag]

            # In distributed FSDP mode, keep one forward per token-step on every rank
            # so collectives stay aligned even if this rank has no active samples.
            if not active_idx:
                if self.world_size > 1:
                    dummy = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _ = self.model(dummy, last_token_only=True)
                    continue
                break

            logits_chunks = []
            micro_bsz = max(1, int(self.rollout_micro_bsz))
            for st in range(0, len(active_idx), micro_bsz):
                sub_idx = active_idx[st: st + micro_bsz]
                seqs = [full_tokens[i][-self.ctx_len:] for i in sub_idx]
                inp, lens = self._pad_batch(seqs, pad_id=0)
                same_len = all(int(l) == int(lens[0]) for l in lens)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if same_len:
                        logits_sub = self.model(inp, last_token_only=True)
                        if torch.is_tensor(logits_sub) and logits_sub.dim() == 3:
                            logits_sub = logits_sub[:, -1, :]
                    else:
                        logits_full = self.model(inp)
                        if torch.is_tensor(logits_full) and logits_full.dim() == 2:
                            logits_full = logits_full.unsqueeze(0)
                        take_idx = torch.tensor(
                            [max(0, int(l) - 1) for l in lens],
                            dtype=torch.long,
                            device=inp.device,
                        )
                        batch_idx = torch.arange(inp.size(0), dtype=torch.long, device=inp.device)
                        logits_sub = logits_full[batch_idx, take_idx, :]
                logits_chunks.append(logits_sub)
            logits_last = torch.cat(logits_chunks, dim=0) if len(logits_chunks) > 1 else logits_chunks[0]

            tok_local, lp_local = _sample_from_logits(logits_last, active_idx)
            tok_cpu = tok_local.detach().cpu().tolist()
            lp_cpu = lp_local.detach().cpu().tolist()

            for li, gi in enumerate(active_idx):
                if not active[gi]:
                    continue
                tok = int(tok_cpu[li])
                lp = float(lp_cpu[li])
                full_tokens[gi].append(tok)
                comp_tokens[gi].append(tok)
                old_logps[gi].append(lp)

                if apply_penalty:
                    cnt = rep_counts[gi]
                    if penalty_decay not in (0.0, 1.0):
                        for k in list(cnt.keys()):
                            nv = float(cnt[k]) * float(penalty_decay)
                            if nv < 1e-5:
                                del cnt[k]
                            else:
                                cnt[k] = nv
                    cnt[tok] = float(cnt.get(tok, 0.0) + 1.0)

                if tok in STOP_TOKENS:
                    active[gi] = False

            if (stop_on_user or stop_on_boxed) and (t % stop_check_every == 0):
                for gi in active_idx:
                    if not active[gi]:
                        continue
                    w = comp_tokens[gi][-stop_check_window:] if stop_check_window > 0 else comp_tokens[gi]
                    s = self.decode(w)
                    if stop_on_boxed and boxed_complete(s):
                        active[gi] = False
                        continue
                    if stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[gi] = False
                        continue

        for i in range(B):
            if active[i]:
                truncated[i] = True

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_tokens, old_logps, comp_text, truncated


# =========================================================
# Trainer - full-parameter GRPO
# =========================================================

class GRPOFullFinetuneTrainer:
    def __init__(
        self,
        train_model,
        infer_engine,
        encode_fn,
        decode_fn,
        data: List[Dict[str, Any]],
        judge_client: FileJudgeClient,
        out_dir: str,
        device: str,
        cfg: GRPOConfig,
        seed: int = 42,
        eval_data: List[Dict[str, Any]] = None,
        train_gpu_count: int = 1,
        rank: int = 0,
        world_size: int = 1,
        model_pth_path: str = "",
        model_init_path: str = "",
    ):
        self.model = train_model
        self.base_model = _unwrap_model(train_model)
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.data = data
        self.eval_data = eval_data if eval_data is not None else data  # Use eval_data if provided
        self.judge_client = judge_client
        self.out_dir = out_dir
        self.device = device
        self.device_index = _parse_cuda_index(device)
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.rollout_rng = random.Random(int(seed) + 100003 * int(rank + 1))
        self.train_gpu_count = max(1, int(train_gpu_count))
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.is_main = (self.rank == 0)
        self.model_pth_path = str(model_pth_path or "")
        self.model_init_path = str(model_init_path or "")
        self.fsdp_enabled = (FSDP is not None) and isinstance(self.model, FSDP)
        self.write_outputs = self.is_main

        # Fixed eval indices to keep eval deterministic across training
        self._fixed_eval_indices = None

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.rank_log_path = os.path.join(out_dir, f"train_rank{self.rank}.log")
        self.train_gen_dump_path = os.path.join(out_dir, "train_gen_judgements.jsonl")
        self.eval_gen_dump_path = os.path.join(out_dir, "eval_gen_judgements.jsonl")
        self.infer_check_path = os.path.join(out_dir, "infer_check.jsonl")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")

        self._hang_f = None
        ts0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        boot_line = f"[{ts0}] [BOOT] rank={self.rank} world_size={self.world_size} pid={os.getpid()} device={self.device}"
        try:
            _append_text_line(self.rank_log_path, boot_line)
        except Exception as e:
            print(f"[rank-log-error] rank={self.rank} path={self.rank_log_path} err={e}", file=sys.stderr, flush=True)
        if self.is_main:
            try:
                _append_text_line(self.log_path, boot_line)
            except Exception as e:
                print(f"[main-log-error] path={self.log_path} err={e}", file=sys.stderr, flush=True)
        if cfg.enable_faulthandler:
            try:
                import faulthandler
                hang_path = os.path.join(out_dir, "hang_tracebacks.log")
                self._hang_f = open(hang_path, "a", encoding="utf-8", buffering=1)
                faulthandler.enable(file=self._hang_f, all_threads=True)
                if float(cfg.hang_dump_s) > 0:
                    faulthandler.dump_traceback_later(float(cfg.hang_dump_s), repeat=True, file=self._hang_f)
            except Exception:
                self._hang_f = None

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable params found.")

        if self.cfg.memory_efficient_adamw:
            self.opt = MemoryEfficientAdamW(
                params,
                lr=self.cfg.lr,
                betas=(float(self.cfg.beta1), float(self.cfg.beta2)),
                eps=float(self.cfg.optimizer_eps),
                weight_decay=0.0,
                enabled=True,
            )
            self._log("Optimizer: MemoryEfficientAdamW (CPU-offloaded states)")
        else:
            self.opt = torch.optim.Adam(
                params,
                lr=self.cfg.lr,
                betas=(float(self.cfg.beta1), float(self.cfg.beta2)),
                eps=float(self.cfg.optimizer_eps),
                weight_decay=0.0,
            )

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        if self.is_main:
            print(line, flush=True)
        try:
            _append_text_line(self.rank_log_path, line)
        except Exception as e:
            print(f"[rank-log-error] rank={self.rank} path={self.rank_log_path} err={e}", file=sys.stderr, flush=True)
        if self.is_main:
            try:
                _append_text_line(self.log_path, line)
            except Exception as e:
                print(f"[main-log-error] path={self.log_path} err={e}", file=sys.stderr, flush=True)

    def _judge_api_batch(self, items: List[Dict[str, Any]], tag: str) -> Dict[str, Dict[str, Any]]:
        return self.judge_client.judge(items, tag=tag)

    def _build_api_judge_debug(self, gt: str, result: Dict[str, Any]) -> Dict[str, Any]:
        ok = bool(result.get("ok"))
        return {
            "gt": gt,
            "method": "local_regex_verl",
            "raw_extracted_answer": result.get("raw"),
            "extract_source": result.get("extract_source"),
            "pred_norm": result.get("pred_norm"),
            "gt_norm": result.get("gt_norm"),
            "error": result.get("error"),
            "correct": ok,
        }

    def _is_api_error(self, result: Optional[Dict[str, Any]]) -> bool:
        if result is None:
            return True
        return bool(result.get("error"))

    def _detect_world_size(self) -> int:
        return self.world_size

    def _set_train_device_for_dist(self):
        if self.device_index is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.device_index)

    def _dist_barrier(self):
        if self.world_size > 1 and _dist_is_initialized():
            self._set_train_device_for_dist()
            if torch.cuda.is_available() and self.device_index is not None:
                dist.barrier(device_ids=[int(self.device_index)])
            else:
                dist.barrier()

    def _collect_model_state_for_save(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.fsdp_enabled:
            if FSDP is None or StateDictType is None or FullStateDictConfig is None:
                raise RuntimeError("FSDP state-dict helpers are unavailable in current torch build.")
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_cfg):
                state = self.model.state_dict()
            if not self.is_main:
                return None
            return state
        return {n: p.detach().cpu() for n, p in self.base_model.named_parameters()}

    @torch.no_grad()
    def _sync_infer_weights(self, step: int, force: bool = False):
        if self.infer is None:
            return
        if isinstance(self.infer, TrainModelRolloutEngine):
            return

        if self.fsdp_enabled and self.world_size > 1:
            do_sync = True
            if hasattr(self.infer, "should_sync"):
                do_sync = bool(self.infer.should_sync(step=step, force=force))
            if self.world_size > 1 and _dist_is_initialized():
                self._set_train_device_for_dist()
                flag = [1 if do_sync else 0]
                dist.broadcast_object_list(flag, src=0)
                do_sync = bool(flag[0])
            if not do_sync:
                self._dist_barrier()
                return

            if FSDP is None:
                raise RuntimeError("FSDP state-dict helpers are unavailable in current torch build.")

            self._set_train_device_for_dist()
            base_model = self.base_model
            n_layers = int(getattr(getattr(base_model, "args", None), "n_layer", 0))
            offload_to_cpu = bool(getattr(self.cfg, "sync_infer_offload_cpu", False))
            t_sync = time.time()

            try:
                ctx = FSDP.summon_full_params(
                    self.model,
                    recurse=True,
                    writeback=False,
                    rank0_only=False,
                    offload_to_cpu=offload_to_cpu,
                )
            except TypeError:
                ctx = FSDP.summon_full_params(self.model, recurse=True, writeback=False)

            def _getter(name: str):
                return _get_tensor_by_dotted_name(base_model, name)

            with ctx:
                self.infer.sync_infer_weights(
                    step=step,
                    force=force,
                    train_tensor_getter=_getter,
                    n_layers=n_layers,
                )
            if self.is_main:
                self._log(
                    f"[sync] train->rollout done: step={step} "
                    f"offload_cpu={int(offload_to_cpu)} dt={time.time() - t_sync:.2f}s"
                )
            self._dist_barrier()
            return

        self.infer.sync_infer_weights(step=step, force=force)

    def _model_stats(self):
        mx = 0.0
        rms_sum = 0.0
        cnt = 0
        bad = False
        with torch.no_grad():
            for _, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.numel() == 0:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad = True
                v = p.detach().float()
                mx = max(mx, float(v.abs().max().item()))
                rms_sum += float((v * v).mean().sqrt().item())
                cnt += 1
        return {"absmax": mx, "rms_avg": (rms_sum / max(1, cnt)), "bad": bad}

    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
        lens = [len(s) for s in seqs]
        T = max(lens)
        B = len(seqs)
        x = torch.full((B, T), pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return x, lens

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return (rewards - mean) / (std + float(self.cfg.advantage_eps))

    def _ppo_clipped_objective(self, ratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv
        return torch.minimum(unclipped, clipped)

    @torch.no_grad()
    def _infer_once(self, problem: str, gt: str, max_new: int, temperature: float, top_p: float, top_k: int, tag: str) -> Dict[str, Any]:
        if self.infer is None:
            raise RuntimeError("Inference engine is unavailable on this rank.")
        prompt = build_prompt(problem)
        ids = self.encode(prompt)
        max_prompt_len = int(self.base_model.args.ctx_len) - int(max_new) - 4
        max_prompt_len = max(64, max_prompt_len)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]

        comp_tokens, _, comp_texts, truncs = self.infer.generate_group_parallel(
            prompt_tokens_list=[ids],
            group_size=1,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=self.cfg.stop_on_think_close,
            stop_on_user=self.cfg.stop_on_user,
            stop_on_boxed=self.cfg.stop_on_boxed,
            stop_check_every=max(1, self.cfg.stop_check_every // 2),
            stop_check_window=max(64, self.cfg.stop_check_window),
        )

        txt = comp_texts[0]
        trunc = bool(truncs[0])
        item_id = "infer_0"
        results = self._judge_api_batch(
            [{"item_id": item_id, "pred": txt, "gt": gt, "truncated": trunc}],
            tag=tag,
        )
        res = results.get(item_id, {"ok": False, "error": "missing result", "raw": None})
        if self._is_api_error(res):
            r = None
        else:
            r = 1.0 if res.get("ok") else 0.0
        jdbg = self._build_api_judge_debug(gt, res)
        jdbg["truncated_forced_zero"] = bool(res.get("truncated_forced_zero", False))
        jdbg["api_error"] = self._is_api_error(res)

        return {
            "prompt": prompt,
            "completion": txt,
            "truncated": trunc,
            "reward": (float(r) if r is not None else None),
            "judge": jdbg,
            "gen_len": len(comp_tokens[0]),
        }

    @torch.no_grad()
    def _diag_teacher_forced_stats(
        self,
        trajs: List[Dict[str, Any]],
        token_budget: int,
        prefix_lens: Tuple[int, ...] = (1, 4, 8, 16, 32, 64),
    ) -> Dict[str, float]:
        def _pack(trajs_local: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            if token_budget <= 0:
                return [[tr] for tr in trajs_local]
            batches_local: List[List[Dict[str, Any]]] = []
            i = 0
            while i < len(trajs_local):
                used = 0
                batch_local: List[Dict[str, Any]] = []
                while i < len(trajs_local):
                    tr = trajs_local[i]
                    need = max(1, len(tr["full_tokens"]) - 1)
                    if batch_local and (used + need) > token_budget:
                        break
                    batch_local.append(tr)
                    used += need
                    i += 1
                if not batch_local:
                    batch_local = [trajs_local[i]]
                    i += 1
                batches_local.append(batch_local)
            return batches_local

        old_sum = 0.0
        new_sum = 0.0
        tok_n = 0
        traj_n = 0
        prefix_old_sum = {int(k): 0.0 for k in prefix_lens}
        prefix_new_sum = {int(k): 0.0 for k in prefix_lens}
        prefix_tok_n = {int(k): 0 for k in prefix_lens}

        self.model.eval()
        for batch in (_pack(trajs) if trajs else []):
            seqs = [b["full_tokens"] for b in batch]
            padded, lens = self._pad_batch(seqs, pad_id=0)
            inp = padded[:, :-1].contiguous()
            tgt = padded[:, 1:].contiguous()

            if self.device_index is not None:
                torch.cuda.set_device(self.device_index)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(inp)
            if torch.is_tensor(logits) and logits.dim() == 2:
                logits = logits.unsqueeze(0)

            picked = -F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            ).reshape_as(tgt)

            for bi, tr in enumerate(batch):
                prompt_len = int(tr["prompt_len"])
                comp_len = int(tr["comp_len"])
                full_len = int(lens[bi])
                start = max(0, prompt_len - 1)
                end = max(start, min(full_len - 1, start + comp_len))
                if end <= start:
                    continue

                new_lp = picked[bi, start:end].float()
                keep = min(int(new_lp.numel()), len(tr.get("old_logps", [])))
                if keep <= 0:
                    continue
                new_lp = new_lp[:keep]
                old_lp = torch.tensor(tr["old_logps"][:keep], dtype=torch.float32, device=new_lp.device)

                old_sum += float(old_lp.sum().item())
                new_sum += float(new_lp.sum().item())
                tok_n += keep
                traj_n += 1

                for prefix_len in prefix_lens:
                    take = min(keep, int(prefix_len))
                    if take <= 0:
                        continue
                    prefix_old_sum[int(prefix_len)] += float(old_lp[:take].sum().item())
                    prefix_new_sum[int(prefix_len)] += float(new_lp[:take].sum().item())
                    prefix_tok_n[int(prefix_len)] += int(take)

        if self.world_size > 1 and _dist_is_initialized():
            stats = {
                "old_sum": old_sum,
                "new_sum": new_sum,
                "tok_n": float(tok_n),
                "traj_n": float(traj_n),
            }
            for prefix_len in prefix_lens:
                stats[f"prefix_old_sum_{int(prefix_len)}"] = prefix_old_sum[int(prefix_len)]
                stats[f"prefix_new_sum_{int(prefix_len)}"] = prefix_new_sum[int(prefix_len)]
                stats[f"prefix_tok_n_{int(prefix_len)}"] = float(prefix_tok_n[int(prefix_len)])
            for key, val in list(stats.items()):
                t = torch.tensor(float(val), dtype=torch.float64, device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                stats[key] = float(t.item())
            old_sum = float(stats["old_sum"])
            new_sum = float(stats["new_sum"])
            tok_n = int(round(stats["tok_n"]))
            traj_n = int(round(stats["traj_n"]))
            for prefix_len in prefix_lens:
                prefix_old_sum[int(prefix_len)] = float(stats[f"prefix_old_sum_{int(prefix_len)}"])
                prefix_new_sum[int(prefix_len)] = float(stats[f"prefix_new_sum_{int(prefix_len)}"])
                prefix_tok_n[int(prefix_len)] = int(round(stats[f"prefix_tok_n_{int(prefix_len)}"]))

        if tok_n > 0:
            old_full_mean = old_sum / float(tok_n)
            new_full_mean = new_sum / float(tok_n)
            delta_full_mean = (new_sum - old_sum) / float(tok_n)
        else:
            old_full_mean = float("nan")
            new_full_mean = float("nan")
            delta_full_mean = float("nan")

        out: Dict[str, float] = {
            "traj_n": float(traj_n),
            "tok_n": float(tok_n),
            "old_full_mean": old_full_mean,
            "new_full_mean": new_full_mean,
            "delta_full_mean": delta_full_mean,
        }
        for prefix_len in prefix_lens:
            cnt = int(prefix_tok_n[int(prefix_len)])
            if cnt > 0:
                old_mean = prefix_old_sum[int(prefix_len)] / float(cnt)
                new_mean = prefix_new_sum[int(prefix_len)] / float(cnt)
                delta_mean = new_mean - old_mean
            else:
                old_mean = float("nan")
                new_mean = float("nan")
                delta_mean = float("nan")
            out[f"old_p{int(prefix_len)}_mean"] = old_mean
            out[f"new_p{int(prefix_len)}_mean"] = new_mean
            out[f"delta_p{int(prefix_len)}_mean"] = delta_mean
        return out

    def _pick_diag_grad_params(self, max_n: int) -> List[Tuple[str, torch.nn.Parameter]]:
        items = [
            (n, p)
            for n, p in self.base_model.named_parameters()
            if p.requires_grad and int(p.numel()) > 0 and int(p.numel()) <= 2_000_000
        ]
        if not items:
            items = [(n, p) for n, p in self.base_model.named_parameters() if p.requires_grad and int(p.numel()) > 0]
        if not items:
            return []
        max_n = max(1, int(max_n))
        if len(items) <= max_n:
            return items
        if max_n == 1:
            return [items[len(items) // 2]]
        idxs: List[int] = []
        for i in range(max_n):
            idx = int(round(i * (len(items) - 1) / float(max_n - 1)))
            if idxs and idx == idxs[-1]:
                continue
            idxs.append(idx)
        picked = [items[i] for i in idxs]
        if len(picked) > max_n:
            picked = picked[:max_n]
        return picked

    def _diag_build_local_grad_shard_payload(
        self,
        selected_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        local_param_map = dict(self.base_model.named_parameters())
        shard_payload: Dict[str, Dict[str, Any]] = {}
        for name in selected_names:
            pp = local_param_map.get(name)
            gg = None if pp is None else pp.grad
            shard_payload[name] = {
                "param_local_shape": (tuple(int(x) for x in pp.shape) if pp is not None else tuple()),
                "param_local_numel": (int(pp.numel()) if pp is not None else 0),
                "grad_local_numel": (int(gg.numel()) if gg is not None else 0),
                "grad_flat": (gg.detach().float().reshape(-1).cpu().clone() if gg is not None else None),
            }
        return shard_payload

    def _diag_rebuild_dist_grads_from_gathered(
        self,
        selected_names: List[str],
        gathered_shards: List[Optional[Dict[str, Dict[str, Any]]]],
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Dict[str, Any]]]:
        dist_grads: Dict[str, Optional[torch.Tensor]] = {}
        dist_meta: Dict[str, Dict[str, Any]] = {}
        for name in selected_names:
            chunks = []
            shard_nums = []
            shard_shapes = []
            for rr in range(self.world_size):
                rec = gathered_shards[rr] if rr < len(gathered_shards) else None
                item = rec.get(name) if isinstance(rec, dict) else None
                if not isinstance(item, dict):
                    shard_nums.append(0)
                    shard_shapes.append(tuple())
                    continue
                gflat = item.get("grad_flat")
                gnum = int(item.get("grad_local_numel", 0))
                shard_nums.append(gnum)
                shard_shapes.append(tuple(int(x) for x in item.get("param_local_shape", tuple())))
                if isinstance(gflat, torch.Tensor) and gnum > 0:
                    chunks.append(gflat.reshape(-1).cpu().clone())
            dist_grads[name] = torch.cat(chunks, dim=0).cpu().clone() if chunks else None
            dist_meta[name] = {
                "shard_nums": shard_nums,
                "shard_shapes": shard_shapes,
                "total_shard_numel": int(sum(shard_nums)),
            }
        return dist_grads, dist_meta

    def _diag_collect_selected_dist_grads(
        self,
        selected_names: List[str],
    ) -> Tuple[Optional[Dict[str, Optional[torch.Tensor]]], Optional[Dict[str, Dict[str, Any]]]]:
        if not (self.world_size > 1 and _dist_is_initialized() and self.fsdp_enabled and FSDP is not None):
            return None, None
        shard_payload = self._diag_build_local_grad_shard_payload(selected_names)
        gathered_shards: List[Optional[Dict[str, Dict[str, Any]]]] = [None for _ in range(self.world_size)]
        self._set_train_device_for_dist()
        dist.all_gather_object(gathered_shards, shard_payload)

        if not self.is_main:
            return None, None
        return self._diag_rebuild_dist_grads_from_gathered(selected_names, gathered_shards)

    def _diag_compare_distributed_vs_oracle_grad(
        self,
        *,
        phase: str,
        step: int,
        epoch_idx: int,
        merged_trajs: List[Dict[str, Any]],
        local_batches: List[List[Dict[str, Any]]],
        real_batch_tok_sizes: List[int],
        token_budget: int,
        sync_batch_n: int,
        local_batch_n: int,
        pad_zero: int,
        selected_names: Optional[List[str]] = None,
        prepad_local_payload: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not (self.world_size > 1 and _dist_is_initialized() and self.fsdp_enabled and FSDP is not None):
            return
        if not bool(self.cfg.diag_compare_global_grad):
            return
        if int(step) != int(self.cfg.diag_compare_step):
            return
        if int(epoch_idx) != int(self.cfg.diag_compare_epoch):
            return
        if int(step) != 1 or int(epoch_idx) != 1:
            self._log(f"[GRAD-CMP {phase} step {step}] skip: current dump path only supports step=1 epoch=1")
            return
        if self.model_init_path:
            self._log(f"[GRAD-CMP {phase} step {step}] skip: dump replay currently assumes no model_init override")
            return
        if not self.model_pth_path:
            self._log(f"[GRAD-CMP {phase} step {step}] skip: empty model_pth_path")
            return

        global_trajs = sorted(list(merged_trajs), key=lambda x: len(x["full_tokens"]), reverse=True)
        if bool(self.cfg.online_correct_only_ce):
            global_trajs = [tr for tr in global_trajs if bool(tr.get("is_correct", False))]
        global_tok = int(sum(int(tr.get("comp_len", 0)) for tr in global_trajs))
        if global_tok <= 0:
            self._log(f"[GRAD-CMP {phase} step {step}] skip: global_tok=0")
            return

        if selected_names is None:
            selected = self._pick_diag_grad_params(int(self.cfg.diag_compare_param_count))
            if not selected:
                self._log(f"[GRAD-CMP {phase} step {step}] skip: no trainable params selected")
                return
            selected_names = [name for name, _ in selected]
        else:
            selected_names = [str(name) for name in selected_names]
            if not selected_names:
                self._log(f"[GRAD-CMP {phase} step {step}] skip: empty selected_names")
                return

        def _diag_traj_minimal(tr: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "problem_key": tr.get("problem_key", ""),
                "full_tokens": list(int(x) for x in tr.get("full_tokens", [])),
                "prompt_len": int(tr.get("prompt_len", 0)),
                "comp_len": int(tr.get("comp_len", 0)),
                "old_logps": [float(x) for x in tr.get("old_logps", [])],
                "adv": float(tr.get("adv", 0.0)),
                "is_correct": bool(tr.get("is_correct", False)),
            }

        local_rank_batches = [
            [_diag_traj_minimal(tr) for tr in batch]
            for batch in local_batches
            if batch
        ]
        local_rank_payload = {
            "rank": int(self.rank),
            "local_batch_n": int(local_batch_n),
            "sync_batch_n": int(sync_batch_n),
            "pad_zero": int(pad_zero),
            "real_batch_tok_sizes": [int(x) for x in real_batch_tok_sizes],
            "batches": local_rank_batches,
        }
        gathered_rank_batches: List[Optional[Dict[str, Any]]] = [None for _ in range(self.world_size)]
        self._set_train_device_for_dist()
        dist.all_gather_object(gathered_rank_batches, local_rank_payload)

        dist_grads, dist_meta = self._diag_collect_selected_dist_grads(selected_names)
        prepad_dist_grads = None
        prepad_dist_meta = None
        if prepad_local_payload is not None or self.world_size > 1:
            gathered_prepad = ([None for _ in range(self.world_size)] if self.is_main else None)
            self._set_train_device_for_dist()
            dist.gather_object((prepad_local_payload if prepad_local_payload is not None else {}), gathered_prepad, dst=0)
            if self.is_main and isinstance(gathered_prepad, list):
                prepad_dist_grads, prepad_dist_meta = self._diag_rebuild_dist_grads_from_gathered(selected_names, gathered_prepad)

        if not self.is_main:
            self._dist_barrier()
            return

        dump_dir = os.path.join(self.out_dir, '_diag_gradcmp')
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f'{phase}_step{int(step)}_epoch{int(epoch_idx)}_dump.pt')
        payload = {
            'phase': str(phase),
            'step': int(step),
            'epoch_idx': int(epoch_idx),
            'model_pth_path': str(self.model_pth_path),
            'ctx_len': int(getattr(getattr(self.base_model, 'args', None), 'ctx_len', 0)),
            'grad_cp': int(getattr(getattr(self.base_model, 'args', None), 'grad_cp', 0)),
            'token_budget': int(token_budget),
            'sync_batch_n': int(sync_batch_n),
            'local_batch_n': int(local_batch_n),
            'pad_zero': int(pad_zero),
            'global_tok': int(global_tok),
            'world_size': int(self.world_size),
            'selected_names': list(selected_names),
            'dist_grads': dist_grads,
            'dist_meta': dist_meta,
            'dist_grads_prepad': prepad_dist_grads,
            'dist_meta_prepad': prepad_dist_meta,
            'trajs': global_trajs,
            'rank_batches': gathered_rank_batches,
            'cfg': {
                'online_correct_only_ce': bool(self.cfg.online_correct_only_ce),
                'use_ppo_loss': bool(self.cfg.use_ppo_loss),
                'clip_range': float(self.cfg.clip_range),
                'entropy_coef': float(self.cfg.entropy_coef),
            },
        }
        torch.save(payload, dump_path)
        self._log(
            f"[GRAD-CMP-DUMP {phase} step {step}] saved={dump_path} trajs={len(global_trajs)} global_tok={global_tok} "
            f"selected={len(selected_names)} batches(local/sync/pad)={int(local_batch_n)}/{int(sync_batch_n)}/{int(pad_zero)}"
        )
        self._dist_barrier()

    @torch.no_grad()
    def sanity_infer_check(self, step: int, n: int = 3):
        for _ in range(n):
            ex = self.data[self.rng.randrange(len(self.data))]
            rec = self._infer_once(
                problem=ex.get("problem", ""),
                gt=str(ex.get("answer", ex.get("solution", ""))),
                max_new=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                tag=f"infer_check_step{step}",
            )
            append_jsonl(self.infer_check_path, {
                "time": now_str(),
                "step": step,
                "problem": ex.get("problem", ""),
                "gt": str(ex.get("answer", ex.get("solution", ""))),
                **rec
            })

    @torch.no_grad()
    def evaluate(self, step: int):
        """Evaluate fixed eval set. Supports both single-rank and distributed sharded eval."""
        if self.infer is None:
            raise RuntimeError("Inference engine is unavailable on this rank.")
        # Initialize fixed eval indices on first call (ensures same questions every eval)
        if self._fixed_eval_indices is None:
            # Use sequential indices instead of random sampling to avoid duplicates
            n_eval = min(self.cfg.eval_n, len(self.eval_data))
            self._fixed_eval_indices = list(range(n_eval))
            self._log(f"[EVAL] Initialized fixed eval indices: {len(self._fixed_eval_indices)} questions (sequential, no duplicates)")

        idxs = self._fixed_eval_indices
        ws = max(1, int(self.world_size))
        if ws > 1:
            local_pos = list(range(self.rank, len(idxs), ws))
            local_idxs = [idxs[i] for i in local_pos]
        else:
            local_idxs = list(idxs)

        # Build prompts for local shard
        ex_list = [self.eval_data[i] for i in local_idxs]
        prompt_strs = [build_prompt(ex.get("problem", "")) for ex in ex_list]
        gts = [str(ex.get("answer", ex.get("solution", ""))) for ex in ex_list]

        # Encode prompts
        max_prompt_len = int(self.base_model.args.ctx_len) - int(self.cfg.eval_max_new_tokens) - 4
        max_prompt_len = max(64, max_prompt_len)

        prompt_tokens_list = []
        for ps in prompt_strs:
            ids = self.encode(ps)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens_list.append(ids)

        # Batch inference in chunks of 192
        eval_batch_size = 192
        comp_tokens = []
        comp_texts = []
        truncated = []

        for start in range(0, len(prompt_tokens_list), eval_batch_size):
            batch_prompts = prompt_tokens_list[start:start + eval_batch_size]
            batch_tokens, _, batch_texts, batch_trunc = self.infer.generate_group_parallel(
                prompt_tokens_list=batch_prompts,
                group_size=1,
                max_new_tokens=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.eval_temperature,
                top_p=self.cfg.eval_top_p,
                top_k=self.cfg.eval_top_k,
                stop_on_think_close=self.cfg.stop_on_think_close,
                stop_on_user=self.cfg.stop_on_user,
                stop_on_boxed=self.cfg.stop_on_boxed,
                stop_check_every=max(1, self.cfg.stop_check_every // 2),
                stop_check_window=max(64, self.cfg.stop_check_window),
                presence_penalty=self.cfg.eval_presence_penalty,
                frequency_penalty=self.cfg.eval_frequency_penalty,
                penalty_decay=self.cfg.eval_penalty_decay,
            )
            comp_tokens.extend(batch_tokens)
            comp_texts.extend(batch_texts)
            truncated.extend(batch_trunc)

        # Local regex judge
        judge_items = []
        for i, (ctext, gt) in enumerate(zip(comp_texts, gts)):
            judge_items.append({
                "item_id": str(i),
                "pred": ctext,
                "gt": gt,
                "truncated": bool(truncated[i]),
            })

        api_results = {}
        if judge_items:
            self._log(f"[EVAL] Running local regex judge for {len(judge_items)} samples...")
            api_results = self._judge_api_batch(judge_items, tag=f"eval_step{step}")

        api_correct = 0
        api_valid = 0
        judge_err = 0
        trunc_cnt = 0
        lens = []
        details = []

        for i, (ex, ctext, trunc) in enumerate(zip(ex_list, comp_texts, truncated)):
            gt = gts[i]
            if trunc:
                trunc_cnt += 1

            res = api_results.get(str(i), {"ok": False, "error": "missing", "raw": None})
            is_api_error = self._is_api_error(res)
            api_ok = bool(res.get("ok")) if not is_api_error else False
            api_r = (1.0 if api_ok else 0.0) if not is_api_error else None
            api_jdbg = self._build_api_judge_debug(gt, res)

            if is_api_error:
                judge_err += 1
            else:
                api_valid += 1
                if api_ok:
                    api_correct += 1

            lens.append(len(comp_tokens[i]))

            detail = {
                "idx": int(local_idxs[i]),
                "problem": ex.get("problem", ""),
                "gt": gt,
                "completion": ctext,
                "truncated": trunc,
                "api_reward": (float(api_r) if api_r is not None else None),
                "api_error": is_api_error,
                "api_judge": api_jdbg,
                "gen_len": len(comp_tokens[i]),
            }
            details.append(detail)

        api_acc = api_correct / max(1, api_valid)
        trunc_rate = trunc_cnt / max(1, len(ex_list))
        avg_len = sum(lens) / max(1, len(lens))

        eval_summary = {
            "time": now_str(),
            "step": step,
            "eval_n": len(idxs),
            "api_valid_n": api_valid,
            "api_error_n": judge_err,
            "api_acc": api_acc,
            "trunc_rate": trunc_rate,
            "avg_len": avg_len,
            "eval_temperature": self.cfg.eval_temperature,
            "eval_top_p": self.cfg.eval_top_p,
            "eval_top_k": self.cfg.eval_top_k,
            "eval_max_new_tokens": self.cfg.eval_max_new_tokens,
            "world_size": ws,
        }
        eval_outputs = dict(eval_summary)
        eval_outputs["details"] = details

        if ws > 1 and _dist_is_initialized():
            stats = torch.tensor(
                [
                    float(api_correct),
                    float(api_valid),
                    float(judge_err),
                    float(trunc_cnt),
                    float(sum(lens)),
                    float(len(ex_list)),
                ],
                dtype=torch.float64,
                device=self.device,
            )
            gathered_details = [None for _ in range(ws)]
            self._set_train_device_for_dist()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            dist.all_gather_object(gathered_details, details)
            g_api_correct = int(stats[0].item())
            g_api_valid = int(stats[1].item())
            g_judge_err = int(stats[2].item())
            g_trunc_cnt = int(stats[3].item())
            g_len_sum = float(stats[4].item())
            g_n = int(stats[5].item())

            g_api_acc = g_api_correct / max(1, g_api_valid)
            g_trunc_rate = g_trunc_cnt / max(1, g_n)
            g_avg_len = g_len_sum / max(1, g_n)
            merged_details = []
            for rank_details in gathered_details:
                if rank_details:
                    merged_details.extend(rank_details)
            merged_details.sort(key=lambda x: int(x.get("idx", -1)))

            if self.is_main:
                eval_summary_main = dict(eval_summary)
                eval_summary_main.update({
                    "eval_n": int(g_n),
                    "api_valid_n": g_api_valid,
                    "api_error_n": g_judge_err,
                    "api_acc": g_api_acc,
                    "trunc_rate": g_trunc_rate,
                    "avg_len": g_avg_len,
                })
                eval_outputs_main = dict(eval_summary_main)
                eval_outputs_main["details"] = merged_details
                append_jsonl(self.eval_path, eval_summary_main)
                append_jsonl(self.eval_gen_dump_path, eval_outputs_main)
                self._log(
                    f"[EVAL step {step}] judge_acc={g_api_acc:.3f} valid={g_api_valid} "
                    f"judge_err={g_judge_err} trunc={g_trunc_rate:.3f} avg_len={g_avg_len:.1f}"
                )
            self._dist_barrier()
            return

        append_jsonl(self.eval_path, eval_summary)
        append_jsonl(self.eval_gen_dump_path, eval_outputs)

        self._log(
            f"[EVAL step {step}] judge_acc={api_acc:.3f} valid={api_valid} "
            f"judge_err={judge_err} trunc={trunc_rate:.3f} avg_len={avg_len:.1f}"
        )

    def train(self, total_steps: int):
        def _run_one_step(
            step: int,
            total_steps_local: int,
            phase: str,
            fixed_batch: Optional[List[Dict[str, Any]]] = None,
            run_periodic_hooks: bool = True,
            update_policy: bool = True,
            record_outputs: bool = True,
            force_sync: Optional[bool] = None,
            rollout_stop_check=None,
        ) -> Dict[str, Any]:
            t0 = time.time()
            self.model.eval()
            sync_force = bool(fixed_batch is not None) if force_sync is None else bool(force_sync)
            self._sync_infer_weights(step=step, force=sync_force)

            trajs: List[Dict[str, Any]] = []
            sample_cnt = 0
            sample_correct = 0
            sample_trunc = 0
            sample_api_error = 0
            all0_cnt = 0
            all1_cnt = 0
            judge_err_groups = 0
            zero_var_groups = 0
            valid_group_cnt = 0
            adv_nonzero_trajs = 0
            adv_total_trajs = 0
            qdbg: Dict[str, Dict[str, Any]] = {}
            output_rows: List[Dict[str, Any]] = []

            if self.infer is not None:
                rollout_rounds = 1 if fixed_batch is not None else max(1, int(self.cfg.max_rollout_rounds))
                for rollout_round in range(1, rollout_rounds + 1):
                    fixed_batch_indices = None
                    if fixed_batch is not None:
                        if self.world_size > 1 and str(phase).startswith("overfit"):
                            fixed_batch_indices = [i for i in range(self.rank, len(fixed_batch), self.world_size)]
                            ex_list = [fixed_batch[i] for i in fixed_batch_indices]
                        else:
                            fixed_batch_indices = list(range(len(fixed_batch)))
                            ex_list = list(fixed_batch)
                        local_prompt_n = len(ex_list)
                    else:
                        if self.world_size > 1:
                            base_n = int(self.cfg.batch_prompts) // int(self.world_size)
                            rem_n = int(self.cfg.batch_prompts) % int(self.world_size)
                            local_prompt_n = int(base_n + (1 if self.rank < rem_n else 0))
                        else:
                            local_prompt_n = int(self.cfg.batch_prompts)
                        ex_list = [self.data[self.rollout_rng.randrange(len(self.data))] for _ in range(local_prompt_n)]

                    prompt_builder = build_overfit_prompt if (fixed_batch is not None and str(phase).startswith("overfit")) else build_prompt
                    prompt_strs = [prompt_builder(ex.get("problem", "")) for ex in ex_list]
                    gts = [str(ex.get("answer", ex.get("solution", ""))) for ex in ex_list]
                    probs = [ex.get("problem", "") for ex in ex_list]

                    prompt_tokens_list = []
                    for ps in prompt_strs:
                        ids = self.encode(ps)
                        max_prompt_len = int(self.base_model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                        max_prompt_len = max(64, max_prompt_len)
                        if len(ids) > max_prompt_len:
                            ids = ids[-max_prompt_len:]
                        prompt_tokens_list.append(ids)

                    run_prompts = prompt_tokens_list
                    run_group_size = int(self.cfg.group_size)
                    if self.world_size > 1 and len(run_prompts) == 0:
                        run_prompts = [[0]]
                        run_group_size = 1

                    if fixed_batch is not None and str(phase).startswith("overfit"):
                        if str(phase).startswith("overfit_probe"):
                            base_rollout_seed = int(getattr(self, "_probe_rollout_seed", self.seed))
                        else:
                            base_rollout_seed = int(getattr(self, "_overfit_rollout_seed", self.seed))
                    else:
                        base_rollout_seed = int(step)

                    if fixed_batch is not None and str(phase).startswith("overfit") and local_prompt_n > 0:
                        comp_tokens_flat = []
                        old_logps_flat = []
                        comp_text_flat = []
                        truncated_flat = []
                        for local_pi, prompt_ids in enumerate(run_prompts):
                            global_pi = int(fixed_batch_indices[local_pi]) if fixed_batch_indices is not None else int(local_pi)
                            rollout_seed = int(base_rollout_seed * 10007 + rollout_round * 1009 + global_pi * 1000003)
                            ctoks_i, oldlp_i, ctext_i, trunc_i = self.infer.generate_group_parallel(
                                prompt_tokens_list=[prompt_ids],
                                group_size=run_group_size,
                                max_new_tokens=self.cfg.max_new_tokens,
                                temperature=self.cfg.temperature,
                                top_p=self.cfg.top_p,
                                top_k=self.cfg.top_k,
                                stop_on_think_close=self.cfg.stop_on_think_close,
                                stop_on_user=self.cfg.stop_on_user,
                                stop_on_boxed=self.cfg.stop_on_boxed,
                                stop_check_every=self.cfg.stop_check_every,
                                stop_check_window=self.cfg.stop_check_window,
                                use_rollout_cache=self.cfg.rollout_use_cache,
                                rng_seed=rollout_seed,
                            )
                            comp_tokens_flat.extend(ctoks_i)
                            old_logps_flat.extend(oldlp_i)
                            comp_text_flat.extend(ctext_i)
                            truncated_flat.extend(trunc_i)
                    else:
                        rollout_seed = int(base_rollout_seed * 10007 + rollout_round + self.rank * 1000003)
                        comp_tokens_flat, old_logps_flat, comp_text_flat, truncated_flat = self.infer.generate_group_parallel(
                            prompt_tokens_list=run_prompts,
                            group_size=run_group_size,
                            max_new_tokens=self.cfg.max_new_tokens,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                            top_k=self.cfg.top_k,
                            stop_on_think_close=self.cfg.stop_on_think_close,
                            stop_on_user=self.cfg.stop_on_user,
                            stop_on_boxed=self.cfg.stop_on_boxed,
                            stop_check_every=self.cfg.stop_check_every,
                            stop_check_window=self.cfg.stop_check_window,
                            use_rollout_cache=self.cfg.rollout_use_cache,
                            rng_seed=rollout_seed,
                        )

                    if local_prompt_n <= 0:
                        continue

                    sample_info = []
                    for pi in range(len(prompt_tokens_list)):
                        pi_samples = []
                        for gi in range(self.cfg.group_size):
                            idx = pi * self.cfg.group_size + gi
                            pi_samples.append(
                                (
                                    comp_tokens_flat[idx],
                                    old_logps_flat[idx],
                                    comp_text_flat[idx],
                                    bool(truncated_flat[idx]),
                                )
                            )
                        sample_info.append(pi_samples)

                    judge_items = []
                    for pi in range(len(prompt_tokens_list)):
                        for gi in range(self.cfg.group_size):
                            _, _, ctext, _trunc = sample_info[pi][gi]
                            judge_items.append({
                                "item_id": f"{pi}_{gi}",
                                "pred": ctext,
                                "gt": gts[pi],
                                "truncated": bool(_trunc),
                            })

                    judge_results = {}
                    if judge_items:
                        judge_results = self._judge_api_batch(judge_items, tag=f"{phase}_step{step}_round{rollout_round}")

                    for pi in range(len(prompt_tokens_list)):
                        rewards = []
                        judges = []
                        group_has_error = False
                        valid_gi = []
                        valid_rewards = []
                        problem_key = str(probs[pi])
                        qrec = qdbg.setdefault(problem_key, {"total": 0, "correct": 0, "trunc": 0, "pred_hist": {}, "wrong_examples": []})

                        for gi in range(self.cfg.group_size):
                            ctoks, _, ctext_gi, trunc = sample_info[pi][gi]
                            item_id = f"{pi}_{gi}"
                            res = judge_results.get(item_id, {"ok": False, "error": "missing result", "raw": None})
                            has_api_error = self._is_api_error(res)
                            if has_api_error:
                                group_has_error = True
                                r = 0.0
                            else:
                                answer_reward = float(res.get("score", 1.0 if res.get("ok") else 0.0))
                                fmt_reward = _format_reward_rwkv(ctext_gi)
                                r = float(answer_reward + float(self.cfg.format_reward_coef) * fmt_reward)
                            jdbg = self._build_api_judge_debug(gts[pi], res)
                            jdbg["truncated_forced_zero"] = bool(res.get("truncated_forced_zero", False))
                            jdbg["api_error"] = has_api_error

                            rewards.append(float(r))
                            judges.append(jdbg)
                            if has_api_error:
                                sample_api_error += 1
                                continue

                            sample_cnt += 1
                            is_correct = bool(judges[gi].get("correct", False))
                            sample_correct += int(is_correct)
                            sample_trunc += int(trunc)
                            valid_gi.append(gi)
                            valid_rewards.append(float(r))

                            qrec["total"] += 1
                            qrec["correct"] += int(is_correct)
                            qrec["trunc"] += int(trunc)
                            pred_key = str(judges[gi].get("pred_norm", judges[gi].get("raw_extracted_answer", "[INVALID]")))
                            qrec["pred_hist"][pred_key] = int(qrec["pred_hist"].get(pred_key, 0)) + 1
                            if (not is_correct) and len(qrec["wrong_examples"]) < 2:
                                qrec["wrong_examples"].append({
                                    "pred": pred_key,
                                    "text": str(ctext_gi)[:240],
                                    "truncated": bool(trunc),
                                })

                        if group_has_error:
                            judge_err_groups += 1
                        else:
                            rsum = float(sum(rewards))
                            if rsum == 0.0:
                                all0_cnt += 1
                            if rsum == float(self.cfg.group_size):
                                all1_cnt += 1

                        adv_map = {gi: 0.0 for gi in range(self.cfg.group_size)}
                        if valid_rewards:
                            valid_group_cnt += 1
                            reward_span = float(max(valid_rewards) - min(valid_rewards))
                            if reward_span <= 1e-12:
                                zero_var_groups += 1
                            adv_vals = self._compute_advantages(
                                torch.tensor(valid_rewards, dtype=torch.float32, device=self.device)
                            ).detach().cpu().tolist()
                            for k, gi in enumerate(valid_gi):
                                adv_map[gi] = float(adv_vals[k])

                        if record_outputs:
                            output_rows.append(
                                {
                                    "time": now_str(),
                                    "phase": phase,
                                    "step": step,
                                    "rollout_round": rollout_round,
                                    "judge_error": group_has_error,
                                    "problem": probs[pi],
                                    "solution": gts[pi],
                                    "prompt": prompt_strs[pi],
                                    "group_size": self.cfg.group_size,
                                    "max_new_tokens": self.cfg.max_new_tokens,
                                    "samples": [
                                        {
                                            "i": gi,
                                            "text": sample_info[pi][gi][2],
                                            "truncated": bool(sample_info[pi][gi][3]),
                                            "reward": float(rewards[gi]),
                                            "adv": float(adv_map[gi]),
                                            "api_error": bool(judges[gi].get("api_error")),
                                            "judge": judges[gi],
                                        }
                                        for gi in range(self.cfg.group_size)
                                    ],
                                }
                            )

                        for gi in valid_gi:
                            comp_tokens, old_logps, _, _ = sample_info[pi][gi]
                            if not comp_tokens or not old_logps:
                                continue
                            keep = min(len(comp_tokens), len(old_logps))
                            if keep <= 0:
                                continue
                            adv_total_trajs += 1
                            if abs(float(adv_map[gi])) > 1e-12:
                                adv_nonzero_trajs += 1
                            if update_policy:
                                trajs.append(
                                    {
                                        "full_tokens": prompt_tokens_list[pi] + comp_tokens[:keep],
                                        "prompt_len": len(prompt_tokens_list[pi]),
                                        "comp_len": keep,
                                        "old_logps": [float(x) for x in old_logps[:keep]],
                                        "adv": float(adv_map[gi]),
                                        "reward": float(rewards[gi]),
                                        "is_correct": bool(judges[gi].get("correct", False)),
                                        "problem_key": problem_key,
                                    }
                                )

                    torch.cuda.empty_cache()

            update_trajs_local = list(trajs)
            if self.world_size > 1 and _dist_is_initialized():
                payload = {
                    "trajs": trajs if update_policy else None,
                    "sample_cnt": sample_cnt,
                    "sample_correct": sample_correct,
                    "sample_trunc": sample_trunc,
                    "sample_api_error": sample_api_error,
                    "all0_cnt": all0_cnt,
                    "all1_cnt": all1_cnt,
                    "judge_err_groups": judge_err_groups,
                    "zero_var_groups": zero_var_groups,
                    "valid_group_cnt": valid_group_cnt,
                    "adv_nonzero_trajs": adv_nonzero_trajs,
                    "adv_total_trajs": adv_total_trajs,
                    "qdbg": qdbg,
                    "output_rows": output_rows if record_outputs else [],
                }

                def _merge_payloads(items):
                    merged_trajs = []
                    merged_sample_cnt = 0
                    merged_sample_correct = 0
                    merged_sample_trunc = 0
                    merged_sample_api_error = 0
                    merged_all0_cnt = 0
                    merged_all1_cnt = 0
                    merged_judge_err_groups = 0
                    merged_zero_var_groups = 0
                    merged_valid_group_cnt = 0
                    merged_adv_nonzero_trajs = 0
                    merged_adv_total_trajs = 0
                    merged_qdbg: Dict[str, Dict[str, Any]] = {}
                    merged_output_rows: List[Dict[str, Any]] = []

                    for g in items:
                        if not isinstance(g, dict):
                            continue
                        g_trajs = g.get("trajs", [])
                        if isinstance(g_trajs, list) and g_trajs:
                            merged_trajs.extend(g_trajs)
                        merged_sample_cnt += int(g.get("sample_cnt", 0))
                        merged_sample_correct += int(g.get("sample_correct", 0))
                        merged_sample_trunc += int(g.get("sample_trunc", 0))
                        merged_sample_api_error += int(g.get("sample_api_error", 0))
                        merged_all0_cnt += int(g.get("all0_cnt", 0))
                        merged_all1_cnt += int(g.get("all1_cnt", 0))
                        merged_judge_err_groups += int(g.get("judge_err_groups", 0))
                        merged_zero_var_groups += int(g.get("zero_var_groups", 0))
                        merged_valid_group_cnt += int(g.get("valid_group_cnt", 0))
                        merged_adv_nonzero_trajs += int(g.get("adv_nonzero_trajs", 0))
                        merged_adv_total_trajs += int(g.get("adv_total_trajs", 0))
                        g_output_rows = g.get("output_rows", [])
                        if isinstance(g_output_rows, list) and g_output_rows:
                            merged_output_rows.extend([row for row in g_output_rows if isinstance(row, dict)])

                        g_qdbg = g.get("qdbg", {})
                        if isinstance(g_qdbg, dict):
                            for qk, qr in g_qdbg.items():
                                dst = merged_qdbg.setdefault(qk, {"total": 0, "correct": 0, "trunc": 0, "pred_hist": {}, "wrong_examples": []})
                                dst["total"] += int(qr.get("total", 0))
                                dst["correct"] += int(qr.get("correct", 0))
                                dst["trunc"] += int(qr.get("trunc", 0))
                                for pk, pv in dict(qr.get("pred_hist", {})).items():
                                    dst["pred_hist"][pk] = int(dst["pred_hist"].get(pk, 0)) + int(pv)
                                for item in list(qr.get("wrong_examples", [])):
                                    if len(dst["wrong_examples"]) < 2:
                                        dst["wrong_examples"].append(item)

                    return {
                        "trajs": merged_trajs,
                        "sample_cnt": merged_sample_cnt,
                        "sample_correct": merged_sample_correct,
                        "sample_trunc": merged_sample_trunc,
                        "sample_api_error": merged_sample_api_error,
                        "all0_cnt": merged_all0_cnt,
                        "all1_cnt": merged_all1_cnt,
                        "judge_err_groups": merged_judge_err_groups,
                        "zero_var_groups": merged_zero_var_groups,
                        "valid_group_cnt": merged_valid_group_cnt,
                        "adv_nonzero_trajs": merged_adv_nonzero_trajs,
                        "adv_total_trajs": merged_adv_total_trajs,
                        "qdbg": merged_qdbg,
                        "output_rows": merged_output_rows,
                    }

                self._set_train_device_for_dist()
                if not update_policy:
                    if str(phase).startswith("overfit"):
                        self._log(
                            f"[{phase} step {step}] before file-sync sample_cnt={sample_cnt} "
                            f"trajs={len(trajs)} qdbg={len(qdbg)} update_policy={int(bool(update_policy))}"
                        )
                    sync_dir = os.path.join(self.out_dir, "_dist_sync")
                    os.makedirs(sync_dir, exist_ok=True)
                    sync_nonce = int(getattr(self, "_dist_sync_nonce", 0)) + 1
                    self._dist_sync_nonce = sync_nonce
                    tag = f"{phase}_step{step}_round{rollout_round}_sync{sync_nonce}"
                    local_path = os.path.join(sync_dir, f"{tag}_rank{self.rank}.pkl")
                    merged_path = os.path.join(sync_dir, f"{tag}_merged.pkl")
                    with open(local_path, "wb") as f:
                        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                    self._dist_barrier()
                    if self.is_main:
                        gathered = []
                        for rr in range(self.world_size):
                            with open(os.path.join(sync_dir, f"{tag}_rank{rr}.pkl"), "rb") as f:
                                gathered.append(pickle.load(f))
                        merged = _merge_payloads(gathered)
                        with open(merged_path, "wb") as f:
                            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
                        if str(phase).startswith("overfit"):
                            self._log(f"[{phase} step {step}] after file-sync gathered={len(gathered)}")
                    self._dist_barrier()
                    with open(merged_path, "rb") as f:
                        merged = pickle.load(f)
                    self._dist_barrier()
                else:
                    gathered = [None for _ in range(self.world_size)]
                    if str(phase).startswith("overfit"):
                        self._log(
                            f"[{phase} step {step}] before all_gather_object sample_cnt={sample_cnt} "
                            f"trajs={len(trajs)} qdbg={len(qdbg)} update_policy={int(bool(update_policy))}"
                        )
                    dist.all_gather_object(gathered, payload)
                    if str(phase).startswith("overfit"):
                        self._log(f"[{phase} step {step}] after all_gather_object gathered={len(gathered)}")
                    merged = _merge_payloads(gathered)

                merged_trajs = merged.get("trajs", []) if isinstance(merged, dict) else []
                sample_cnt = int(merged.get("sample_cnt", 0)) if isinstance(merged, dict) else 0
                sample_correct = int(merged.get("sample_correct", 0)) if isinstance(merged, dict) else 0
                sample_trunc = int(merged.get("sample_trunc", 0)) if isinstance(merged, dict) else 0
                sample_api_error = int(merged.get("sample_api_error", 0)) if isinstance(merged, dict) else 0
                all0_cnt = int(merged.get("all0_cnt", 0)) if isinstance(merged, dict) else 0
                all1_cnt = int(merged.get("all1_cnt", 0)) if isinstance(merged, dict) else 0
                judge_err_groups = int(merged.get("judge_err_groups", 0)) if isinstance(merged, dict) else 0
                zero_var_groups = int(merged.get("zero_var_groups", 0)) if isinstance(merged, dict) else 0
                valid_group_cnt = int(merged.get("valid_group_cnt", 0)) if isinstance(merged, dict) else 0
                adv_nonzero_trajs = int(merged.get("adv_nonzero_trajs", 0)) if isinstance(merged, dict) else 0
                adv_total_trajs = int(merged.get("adv_total_trajs", 0)) if isinstance(merged, dict) else 0
                qdbg = merged.get("qdbg", {}) if isinstance(merged, dict) and isinstance(merged.get("qdbg", {}), dict) else {}
                merged_output_rows = merged.get("output_rows", []) if isinstance(merged, dict) and isinstance(merged.get("output_rows", []), list) else []
                if self.write_outputs and record_outputs and merged_output_rows:
                    for row in merged_output_rows:
                        append_jsonl(self.train_gen_dump_path, row)
            elif self.write_outputs and record_outputs and output_rows:
                for row in output_rows:
                    append_jsonl(self.train_gen_dump_path, row)

            do_log = (phase != "train") or (step % self.cfg.log_interval == 0)
            rollout_acc = float(sample_correct / max(1, sample_cnt))
            rollout_trunc = float(sample_trunc / max(1, sample_cnt))
            should_skip_update = False
            skip_reason = None
            if rollout_stop_check is not None:
                try:
                    stop_info = rollout_stop_check(rollout_acc)
                except Exception:
                    stop_info = None
                if stop_info:
                    should_skip_update = True
                    skip_reason = str(stop_info) if isinstance(stop_info, str) else "rollout stop condition met"

            if do_log:
                self._log(
                    f"[{phase} step {step}/{total_steps_local}] rollout "
                    f"samples={sample_cnt} api_err_samples={sample_api_error} all0_groups={all0_cnt}/{max(1,valid_group_cnt)} all1_groups={all1_cnt}/{max(1,valid_group_cnt)} judge_err_groups={judge_err_groups} | "
                    f"zero_var_groups={zero_var_groups}/{max(1,valid_group_cnt)} adv_nz_trajs={adv_nonzero_trajs}/{max(1,adv_total_trajs)} | "
                    f"acc={rollout_acc:.3f} trunc={rollout_trunc:.3f} "
                    f"trajs={len(merged_trajs) if self.world_size > 1 and _dist_is_initialized() else len(update_trajs_local)}"
                )
            if do_log and qdbg:
                q_items = []
                for qk, qr in qdbg.items():
                    try:
                        total_q = int(qr.get("total", 0))
                        correct_q = int(qr.get("correct", 0))
                        trunc_q = int(qr.get("trunc", 0))
                        pred_hist_q = dict(qr.get("pred_hist", {}))
                    except Exception:
                        continue
                    if total_q <= 0:
                        continue
                    top_pred = "[NONE]"
                    top_pred_n = 0
                    if pred_hist_q:
                        top_pred, top_pred_n = max(pred_hist_q.items(), key=lambda kv: (int(kv[1]), str(kv[0])))
                    label = re.sub(r"\s+", " ", str(qk)).strip()[:60]
                    q_items.append({
                        "label": label,
                        "acc": float(correct_q / max(1, total_q)),
                        "correct": correct_q,
                        "total": total_q,
                        "trunc": trunc_q,
                        "top_pred": str(top_pred)[:24],
                        "top_pred_n": int(top_pred_n),
                    })
                q_items.sort(key=lambda x: (x["acc"], -x["total"], -x["trunc"], x["label"]))
                if q_items:
                    self._log(
                        f"[QDBG {phase} step {step}] " + " | ".join(
                            f"acc={item['acc']:.3f} n={item['correct']}/{item['total']} trunc={item['trunc']} top={item['top_pred']}:{item['top_pred_n']} q={item['label']}"
                            for item in q_items[:4]
                        )
                    )
            if should_skip_update:
                self._log(f"[{phase} step {step}] skip update: {skip_reason}")
                self._dist_barrier()
                return {
                    "acc": rollout_acc,
                    "sample_cnt": int(sample_cnt),
                    "trunc": rollout_trunc,
                    "all0_frac": float(all0_cnt / max(1, valid_group_cnt)),
                    "zero_var_ratio": float(zero_var_groups / max(1, valid_group_cnt)),
                    "adv_nz_ratio": float(adv_nonzero_trajs / max(1, adv_total_trajs)),
                    "skipped": True,
                    "skipped_update_due_to_pass": True,
                }
            if not (merged_trajs if self.world_size > 1 and _dist_is_initialized() else update_trajs_local):
                self._log(f"WARN: empty trajectory batch in {phase} step {step}, skip update.")
                self._dist_barrier()
                return {
                    "acc": rollout_acc,
                    "sample_cnt": int(sample_cnt),
                    "trunc": rollout_trunc,
                    "all0_frac": float(all0_cnt / max(1, valid_group_cnt)),
                    "zero_var_ratio": float(zero_var_groups / max(1, valid_group_cnt)),
                    "adv_nz_ratio": float(adv_nonzero_trajs / max(1, adv_total_trajs)),
                    "skipped": True,
                }

            ws = self._detect_world_size()
            local_trajs = sorted(update_trajs_local, key=lambda x: len(x["full_tokens"]), reverse=True)
            if ws > 1 and len(local_trajs) == 0:
                self._log(
                    f"WARN: world_size={ws} but local_trajs=0 on rank={self.rank}; "
                    "using zero-grad padding batches to keep distributed collectives aligned."
                )

            if bool(self.cfg.online_correct_only_ce):
                local_trajs = [tr for tr in local_trajs if bool(tr.get("is_correct", False))]

            selected_qstats_local: Dict[str, Dict[str, Any]] = {}
            for tr in local_trajs:
                qk = str(tr.get("problem_key", "[UNKNOWN]"))
                rec = selected_qstats_local.setdefault(qk, {"traj_cnt": 0, "tok_cnt": 0})
                rec["traj_cnt"] += 1
                rec["tok_cnt"] += int(tr.get("comp_len", 0))

            total_train_tokens = sum(int(tr["comp_len"]) for tr in local_trajs)
            if total_train_tokens <= 0:
                if ws > 1:
                    self._log(
                        "WARN: total_train_tokens=0 on this rank; "
                        "using zero-grad padding batches for distributed alignment."
                    )
                else:
                    self._log("WARN: total_train_tokens=0, skip update.")
                    self._dist_barrier()
                    return {
                        "acc": float(sample_correct / max(1, sample_cnt)),
                        "sample_cnt": int(sample_cnt),
                        "trunc": float(sample_trunc / max(1, sample_cnt)),
                        "all0_frac": float(all0_cnt / max(1, valid_group_cnt)),
                        "zero_var_ratio": float(zero_var_groups / max(1, valid_group_cnt)),
                        "adv_nz_ratio": float(adv_nonzero_trajs / max(1, adv_total_trajs)),
                        "skipped": True,
                    }

            # GRPO token normalization aligned across distributed ranks:
            # Use global token count, then compensate for averaged gradients by dividing by world_size.
            if ws > 1 and _dist_is_initialized():
                tok_sum_t = torch.tensor(float(max(0, total_train_tokens)), dtype=torch.float64, device=self.device)
                self._set_train_device_for_dist()
                dist.all_reduce(tok_sum_t, op=dist.ReduceOp.SUM)
                global_train_tokens = float(tok_sum_t.item())
                if global_train_tokens <= 0:
                    self._log("WARN: global_train_tokens=0, skip update.")
                    self._dist_barrier()
                    return {
                        "acc": float(sample_correct / max(1, sample_cnt)),
                        "sample_cnt": int(sample_cnt),
                        "trunc": float(sample_trunc / max(1, sample_cnt)),
                        "all0_frac": float(all0_cnt / max(1, valid_group_cnt)),
                        "zero_var_ratio": float(zero_var_groups / max(1, valid_group_cnt)),
                        "adv_nz_ratio": float(adv_nonzero_trajs / max(1, adv_total_trajs)),
                        "skipped": True,
                    }
                denom_tokens_base = max(1.0, global_train_tokens / float(ws))
            else:
                denom_tokens_base = max(1.0, float(total_train_tokens))

            selected_qstats = selected_qstats_local
            global_selected_tok = int(total_train_tokens)
            global_selected_traj = int(len(local_trajs))
            if ws > 1 and _dist_is_initialized():
                gathered_sel = [None for _ in range(ws)]
                self._set_train_device_for_dist()
                dist.all_gather_object(gathered_sel, selected_qstats_local)
                merged_sel: Dict[str, Dict[str, Any]] = {}
                global_selected_tok = 0
                global_selected_traj = 0
                for g in gathered_sel:
                    if not isinstance(g, dict):
                        continue
                    for qk, qr in g.items():
                        dst = merged_sel.setdefault(qk, {"traj_cnt": 0, "tok_cnt": 0})
                        add_traj = int(qr.get("traj_cnt", 0))
                        add_tok = int(qr.get("tok_cnt", 0))
                        dst["traj_cnt"] += add_traj
                        dst["tok_cnt"] += add_tok
                        global_selected_traj += add_traj
                        global_selected_tok += add_tok
                selected_qstats = merged_sel

            if do_log:
                if selected_qstats:
                    sel_items = []
                    for qk, qr in selected_qstats.items():
                        label = re.sub(r"\s+", " ", str(qk)).strip()[:60]
                        sel_items.append({
                            "label": label,
                            "traj_cnt": int(qr.get("traj_cnt", 0)),
                            "tok_cnt": int(qr.get("tok_cnt", 0)),
                        })
                    sel_items.sort(key=lambda x: (-x["tok_cnt"], -x["traj_cnt"], x["label"]))
                    self._log(
                        f"[SELQ {phase} step {step}] local_tok={int(total_train_tokens)} local_traj={len(local_trajs)} "
                        f"global_tok={int(global_selected_tok)} global_traj={int(global_selected_traj)} " + " | ".join(
                            f"tok={item['tok_cnt']} traj={item['traj_cnt']} q={item['label']}"
                            for item in sel_items[:4]
                        )
                    )
                else:
                    self._log(
                        f"[SELQ {phase} step {step}] local_tok=0 local_traj=0 global_tok={int(global_selected_tok)} global_traj={int(global_selected_traj)}"
                    )

            last_loss = None
            last_clipfrac = None
            last_grad = None
            last_raw_grad = None
            last_entropy = None
            last_raw_grad_local = None
            last_grad_local = None

            for epoch_idx in range(int(self.cfg.ppo_epochs) if update_policy else 0):
                self.model.train()
                self.rng.shuffle(local_trajs)

                denom_tokens = float(denom_tokens_base)
                correct_only_ce = bool(self.cfg.online_correct_only_ce)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]

                self.opt.zero_grad(set_to_none=True)

                update_qstats_local: Dict[str, Dict[str, Any]] = {}
                token_budget = int(self.cfg.ppo_max_token_len_per_gpu)
                debug_policy_tok_total = 0
                debug_policy_sum_total = 0.0
                debug_real_batch_n = 0
                pad_diag_enabled = bool(self.cfg.diag_zero_padding)
                pad_diag_counter = 0
                pad_diag_limit = 8
                pad_diag_sentinels: List[Tuple[str, torch.nn.Parameter]] = []
                if pad_diag_enabled:
                    try:
                        pad_diag_sentinels = [(n, pp) for n, pp in self.model.named_parameters() if pp.requires_grad][:3]
                    except Exception:
                        pad_diag_sentinels = []

                def _grad_diag_snapshot() -> Dict[str, Any]:
                    total_g2 = 0.0
                    total_absmax = 0.0
                    nonnull = 0
                    nonfinite = 0
                    sentinels: Dict[str, Optional[torch.Tensor]] = {}
                    for name, pp in pad_diag_sentinels:
                        if pp.grad is None:
                            sentinels[name] = None
                        else:
                            sentinels[name] = pp.grad.detach().float().clone()
                    for pp in trainable_params:
                        if pp.grad is None:
                            continue
                        nonnull += 1
                        gg = pp.grad.detach().float()
                        total_g2 += float((gg * gg).sum().item())
                        total_absmax = max(total_absmax, float(gg.abs().max().item()))
                        if not bool(torch.isfinite(gg).all().item()):
                            nonfinite += 1
                    return {
                        "g2": total_g2,
                        "absmax": total_absmax,
                        "nonnull": nonnull,
                        "nonfinite": nonfinite,
                        "sentinels": sentinels,
                    }

                def _pack_batches_by_token_budget(trajs_local: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
                    batches_local: List[List[Dict[str, Any]]] = []
                    i = 0
                    if token_budget <= 0:
                        while i < len(trajs_local):
                            batches_local.append([trajs_local[i]])
                            i += 1
                        return batches_local

                    while i < len(trajs_local):
                        used = 0
                        batch_local: List[Dict[str, Any]] = []
                        while i < len(trajs_local):
                            tr = trajs_local[i]
                            need = max(1, len(tr["full_tokens"]) - 1)
                            if batch_local and (used + need) > token_budget:
                                break
                            batch_local.append(tr)
                            used += need
                            i += 1
                        if not batch_local:
                            batch_local = [trajs_local[i]]
                            i += 1
                        batches_local.append(batch_local)
                    return batches_local

                batches = _pack_batches_by_token_budget(local_trajs)
                real_batch_tok_sizes = [sum(max(1, len(tr["full_tokens"]) - 1) for tr in batch) for batch in batches if batch]
                local_batch_n = int(len(batches))
                sync_batch_n = int(local_batch_n)
                pad_zero = 0
                if ws > 1 and _dist_is_initialized():
                    gathered_batch_n: List[Optional[int]] = [None for _ in range(ws)]
                    self._set_train_device_for_dist()
                    dist.all_gather_object(gathered_batch_n, local_batch_n)
                    valid_batch_n = [int(x) for x in gathered_batch_n if x is not None]
                    sync_batch_n = int(max(valid_batch_n)) if valid_batch_n else local_batch_n

                    pad_zero = max(0, int(sync_batch_n) - int(local_batch_n))
                    if pad_zero > 0:
                        # Keep collective counts aligned across ranks without changing real-sample weights.
                        batches.extend([[] for _ in range(pad_zero)])

                    if local_batch_n != sync_batch_n or pad_zero > 0:
                        self._log(
                            f"[{phase} step {step}] token-budget align batches: local={local_batch_n}, "
                            f"sync={sync_batch_n}, pad_zero={pad_zero}"
                        )

                def _backward_zero_padding():
                    nonlocal pad_diag_counter
                    pre_diag = None
                    if pad_diag_enabled and pad_diag_counter < pad_diag_limit:
                        pre_diag = _grad_diag_snapshot()
                    if self.device_index is not None:
                        torch.cuda.set_device(self.device_index)
                    dummy_full = torch.zeros((1, 2), dtype=torch.long, device=self.device)
                    dummy_inp = dummy_full[:, :-1].contiguous()
                    dummy_tgt = dummy_full[:, 1:].contiguous()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        dummy_logits = self.model(dummy_inp)
                    if torch.is_tensor(dummy_logits) and dummy_logits.dim() == 2:
                        dummy_logits = dummy_logits.unsqueeze(0)
                    dummy_loss = F.cross_entropy(
                        dummy_logits.float().reshape(-1, dummy_logits.size(-1)),
                        dummy_tgt.reshape(-1),
                        reduction="mean",
                    ) * 0.0
                    dummy_loss.backward()
                    if pre_diag is not None:
                        post_diag = _grad_diag_snapshot()
                        sent_parts = []
                        sent_max = 0.0
                        for name, pre_t in pre_diag["sentinels"].items():
                            post_t = post_diag["sentinels"].get(name)
                            if pre_t is None and post_t is None:
                                delta = 0.0
                            elif pre_t is None:
                                delta = float(post_t.abs().max().item())
                            elif post_t is None:
                                delta = float(pre_t.abs().max().item())
                            else:
                                delta = float((post_t - pre_t).abs().max().item())
                            sent_max = max(sent_max, delta)
                            sent_parts.append(f"{name}:{delta:.2e}")
                        pre_grad = math.sqrt(max(0.0, float(pre_diag["g2"])))
                        post_grad = math.sqrt(max(0.0, float(post_diag["g2"])))
                        self._log(
                            f"[PAD-DIAG {phase} step {step}] e{epoch_idx + 1} pad_call={pad_diag_counter + 1} "
                            f"dummy_loss={float(dummy_loss.detach().item()):.2e} "
                            f"pre_grad={pre_grad:.6f} post_grad={post_grad:.6f} "
                            f"pre_absmax={float(pre_diag['absmax']):.2e} post_absmax={float(post_diag['absmax']):.2e} "
                            f"pre_nonnull={int(pre_diag['nonnull'])} post_nonnull={int(post_diag['nonnull'])} "
                            f"pre_nonfinite={int(pre_diag['nonfinite'])} post_nonfinite={int(post_diag['nonfinite'])} "
                            f"sent_max={sent_max:.2e} sent={' | '.join(sent_parts)}"
                        )
                        pad_diag_counter += 1

                diag_selected_names = None
                diag_prepad_local_payload = None
                if (
                    ws > 1 and _dist_is_initialized()
                    and bool(self.cfg.diag_compare_global_grad)
                    and int(step) == int(self.cfg.diag_compare_step)
                    and int(epoch_idx + 1) == int(self.cfg.diag_compare_epoch)
                    and int(step) == 1
                    and int(epoch_idx + 1) == 1
                    and (not self.model_init_path)
                ):
                    diag_selected = self._pick_diag_grad_params(int(self.cfg.diag_compare_param_count))
                    if diag_selected:
                        diag_selected_names = [name for name, _ in diag_selected]

                real_batches = batches[:local_batch_n]
                pad_batches = batches[local_batch_n:]

                for batch in real_batches:
                    debug_real_batch_n += 1
                    seqs = [b["full_tokens"] for b in batch]
                    padded, lens = self._pad_batch(seqs, pad_id=0)
                    inp = padded[:, :-1].contiguous()
                    tgt = padded[:, 1:].contiguous()

                    if self.device_index is not None:
                        torch.cuda.set_device(self.device_index)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = self.model(inp)
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)

                    logits_f = logits.float()
                    picked = -F.cross_entropy(
                        logits_f.reshape(-1, logits_f.size(-1)),
                        tgt.reshape(-1),
                        reduction="none",
                    ).reshape_as(tgt)
                    entropy = None
                    if self.cfg.entropy_coef > 0:
                        logp = F.log_softmax(logits_f, dim=-1)
                        probs = torch.softmax(logits_f, dim=-1)
                        entropy = -(probs * logp).sum(dim=-1)

                    policy_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                    ent_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                    clip_cnt = torch.zeros((), device=self.device, dtype=torch.float32)
                    tok_cnt = 0

                    for bi, tr in enumerate(batch):
                        prompt_len = int(tr["prompt_len"])
                        comp_len = int(tr["comp_len"])
                        full_len = int(lens[bi])
                        start = max(0, prompt_len - 1)
                        end = max(start, min(full_len - 1, start + comp_len))
                        if end <= start:
                            continue

                        new_lp = picked[bi, start:end].float()
                        ent_seg = entropy[bi, start:end].float() if entropy is not None else None

                        if correct_only_ce:
                            policy_sum = policy_sum + (-new_lp.sum())
                        else:
                            old_lp = None
                            if self.cfg.use_ppo_loss:
                                old_lp = torch.tensor(tr["old_logps"][: end - start], device=self.device, dtype=torch.float32)
                                if old_lp.numel() != new_lp.numel():
                                    keep = min(old_lp.numel(), new_lp.numel())
                                    if keep <= 0:
                                        continue
                                    new_lp = new_lp[:keep]
                                    old_lp = old_lp[:keep]
                                    if ent_seg is not None:
                                        ent_seg = ent_seg[:keep]

                            adv_val = float(tr["adv"])
                            adv = torch.full_like(new_lp, adv_val)

                            if self.cfg.use_ppo_loss:
                                ratio = torch.exp((new_lp - old_lp).clamp(min=-20.0, max=20.0))
                                obj = self._ppo_clipped_objective(ratio, adv)
                                policy_sum = policy_sum + (-obj.sum())
                                clip_cnt = clip_cnt + (
                                    (ratio > (1.0 + self.cfg.clip_range)) | (ratio < (1.0 - self.cfg.clip_range))
                                ).float().sum()
                            else:
                                policy_sum = policy_sum + (-(new_lp * adv).sum())

                        qk = str(tr.get("problem_key", "[UNKNOWN]"))
                        qrec = update_qstats_local.setdefault(qk, {"traj_cnt": 0, "tok_cnt": 0, "ce_sum": 0.0})
                        qrec["traj_cnt"] += 1
                        qrec["tok_cnt"] += int(new_lp.numel())
                        qrec["ce_sum"] += float(((-new_lp).sum()).detach().item())

                        if ent_seg is not None:
                            ent_sum = ent_sum + ent_seg.sum()
                        tok_cnt += int(new_lp.numel())

                    if tok_cnt <= 0:
                        if ws > 1 and _dist_is_initialized():
                            # Keep backward/collective counts aligned even when this real batch
                            # has no valid tokens on current rank.
                            _backward_zero_padding()
                        continue

                    debug_policy_tok_total += int(tok_cnt)
                    debug_policy_sum_total += float(policy_sum.detach().item())

                    loss = policy_sum / float(denom_tokens)
                    if self.cfg.entropy_coef > 0:
                        loss = loss - float(self.cfg.entropy_coef) * (ent_sum / float(denom_tokens))

                    loss.backward()

                    last_loss = float(loss.detach().item())
                    last_entropy = float((ent_sum / float(tok_cnt)).detach().item())
                    last_clipfrac = float((clip_cnt / float(tok_cnt)).detach().item()) if (self.cfg.use_ppo_loss and (not correct_only_ce)) else None

                if diag_selected_names is not None and int(pad_zero) > 0:
                    diag_prepad_local_payload = self._diag_build_local_grad_shard_payload(diag_selected_names)

                for batch in pad_batches:
                    if not batch:
                        # Padding micro-batch: run a zero-weight backward so all ranks execute
                        # identical collective patterns in distributed mode.
                        _backward_zero_padding()

                self._diag_compare_distributed_vs_oracle_grad(
                    phase=phase,
                    step=step,
                    epoch_idx=epoch_idx + 1,
                    merged_trajs=(merged_trajs if ws > 1 and _dist_is_initialized() else local_trajs),
                    local_batches=[list(batch) for batch in real_batches],
                    real_batch_tok_sizes=real_batch_tok_sizes,
                    token_budget=token_budget,
                    sync_batch_n=sync_batch_n,
                    local_batch_n=local_batch_n,
                    pad_zero=pad_zero,
                    selected_names=diag_selected_names,
                    prepad_local_payload=diag_prepad_local_payload,
                )

                merged_update_qstats = update_qstats_local
                if ws > 1 and _dist_is_initialized():
                    gathered_update = [None for _ in range(ws)]
                    self._set_train_device_for_dist()
                    dist.all_gather_object(gathered_update, update_qstats_local)
                    merged_update: Dict[str, Dict[str, Any]] = {}
                    for g in gathered_update:
                        if not isinstance(g, dict):
                            continue
                        for qk, qr in g.items():
                            dst = merged_update.setdefault(qk, {"traj_cnt": 0, "tok_cnt": 0, "ce_sum": 0.0})
                            dst["traj_cnt"] += int(qr.get("traj_cnt", 0))
                            dst["tok_cnt"] += int(qr.get("tok_cnt", 0))
                            dst["ce_sum"] += float(qr.get("ce_sum", 0.0))
                    merged_update_qstats = merged_update
                if do_log and merged_update_qstats:
                    upd_items = []
                    for qk, qr in merged_update_qstats.items():
                        tok_n = int(qr.get("tok_cnt", 0))
                        if tok_n <= 0:
                            continue
                        label = re.sub(r"\s+", " ", str(qk)).strip()[:60]
                        ce_sum = float(qr.get("ce_sum", 0.0))
                        upd_items.append({
                            "label": label,
                            "traj_cnt": int(qr.get("traj_cnt", 0)),
                            "tok_cnt": tok_n,
                            "ce_ptok": ce_sum / float(tok_n),
                        })
                    upd_items.sort(key=lambda x: (-x["ce_ptok"], -x["tok_cnt"], x["label"]))
                    if upd_items:
                        self._log(
                            f"[UPDQ {phase} step {step} e{epoch_idx + 1}] " + " | ".join(
                                f"ce/tok={item['ce_ptok']:.4f} tok={item['tok_cnt']} traj={item['traj_cnt']} q={item['label']}"
                                for item in upd_items[:4]
                            )
                        )

                with torch.no_grad():
                    raw_g2 = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            g = p.grad.detach().float()
                            raw_g2 += float((g.norm(2) ** 2).item())
                    raw_g2_local = float(raw_g2)
                    if self.world_size > 1 and _dist_is_initialized():
                        raw_g2_t = torch.tensor(raw_g2, dtype=torch.float64, device=self.device)
                        dist.all_reduce(raw_g2_t, op=dist.ReduceOp.SUM)
                        raw_g2 = float(raw_g2_t.item())
                    last_raw_grad = math.sqrt(raw_g2)
                    last_raw_grad_local = math.sqrt(raw_g2_local)

                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    if self.fsdp_enabled and hasattr(self.model, "clip_grad_norm_"):
                        self.model.clip_grad_norm_(self.cfg.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip)

                with torch.no_grad():
                    g2 = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += float((g.norm(2) ** 2).item())
                    g2_local = float(g2)
                    if self.world_size > 1 and _dist_is_initialized():
                        g2_t = torch.tensor(g2, dtype=torch.float64, device=self.device)
                        dist.all_reduce(g2_t, op=dist.ReduceOp.SUM)
                        g2 = float(g2_t.item())
                    last_grad = math.sqrt(g2)
                    last_grad_local = math.sqrt(g2_local)

                _pre_snap = None
                if step <= 3 and self.is_main:
                    _pre_snap = {n: p.data.detach().clone() for n, p in
                                 zip([n for n, _ in self.model.named_parameters() if _.requires_grad][:3],
                                     [p for p in trainable_params[:3]])}

                self.opt.step()

                if _pre_snap is not None:
                    for n, old_val in _pre_snap.items():
                        new_val = dict(self.model.named_parameters())[n].data
                        diff = (new_val.float() - old_val.float()).abs().max().item()
                        self._log(f"[UPDATE-CHECK] {n}: dtype={new_val.dtype} max_diff={diff:.2e}")

                self.opt.zero_grad(set_to_none=True)

                if bool(self.cfg.diag_inner_update):
                    batch_min = min(real_batch_tok_sizes) if real_batch_tok_sizes else 0
                    batch_max = max(real_batch_tok_sizes) if real_batch_tok_sizes else 0
                    batch_mean = (sum(real_batch_tok_sizes) / max(1, len(real_batch_tok_sizes))) if real_batch_tok_sizes else 0.0
                    self._log(
                        f"[INNER-DIAG {phase} step {step}] epoch={epoch_idx + 1}/{int(self.cfg.ppo_epochs)} "
                        f"local_traj={len(local_trajs)} local_tok={int(total_train_tokens)} denom={float(denom_tokens):.1f} "
                        f"policy_tok={int(debug_policy_tok_total)} policy_sum={float(debug_policy_sum_total):.4f} "
                        f"batches(real/local/sync/pad)={int(debug_real_batch_n)}/{int(local_batch_n)}/{int(sync_batch_n)}/{int(pad_zero)} "
                        f"tok_per_batch(min/mean/max)={batch_min}/{batch_mean:.1f}/{batch_max} "
                        f"loss={last_loss} raw_grad={last_raw_grad} raw_grad_local={last_raw_grad_local} "
                        f"grad={last_grad} grad_local={last_grad_local}"
                    )

            self.model.eval()
            self._dist_barrier()

            avg_r = sample_correct / max(1, sample_cnt)
            dt = time.time() - t0
            st = self._model_stats()
            lr_now = float(self.opt.param_groups[0]["lr"])

            if do_log:
                self._log(
                    f"[{phase} step {step}/{total_steps_local}] "
                    f"samples={sample_cnt} api_err_samples={sample_api_error} all0_groups={all0_cnt}/{max(1,valid_group_cnt)} all1_groups={all1_cnt}/{max(1,valid_group_cnt)} | "
                    f"acc={sample_correct/max(1,sample_cnt):.3f} trunc={sample_trunc/max(1,sample_cnt):.3f} | "
                    f"avg_reward={avg_r:.4f} loss={last_loss} ent={last_entropy} clipfrac={last_clipfrac} raw_grad={last_raw_grad} grad={last_grad} | "
                    f"model(absmax={st['absmax']:.4f}, rms={st['rms_avg']:.8f}, bad={st['bad']}) | "
                    f"hp(lr={lr_now:.2e}, ratio_clip={self.cfg.clip_range}, grad_clip={self.cfg.grad_clip}, ent={self.cfg.entropy_coef}, "
                    f"temp={self.cfg.temperature}, top_p={self.cfg.top_p}, max_new={self.cfg.max_new_tokens}) "
                    f"step_time={dt:.1f}s"
                )

            if run_periodic_hooks:
                if self.world_size == 1 and self.is_main and self.infer is not None and step % self.cfg.infer_check_interval == 0:
                    self.model.eval()
                    self.sanity_infer_check(step=step, n=3)

                if self.infer is not None and step % self.cfg.eval_interval == 0:
                    self.model.eval()
                    self.evaluate(step=step)

                if step % self.cfg.save_interval == 0 or step == total_steps_local:
                    model_state = self._collect_model_state_for_save()
                    if self.write_outputs and model_state is not None:
                        ckpt_path = os.path.join(self.out_dir, f"ckpt_step{step}.pth")
                        torch.save(
                            {
                                "time": now_str(),
                                "step": step,
                                "cfg": self.cfg.__dict__,
                                "model_state": model_state,
                            },
                            ckpt_path,
                        )

                        latest_full_path = os.path.join(self.out_dir, "latest_full_model.pth")
                        torch.save(model_state, latest_full_path)
                        self._log(f"saved: {ckpt_path}")
                        self._log(f"saved: {latest_full_path}")
                    self._dist_barrier()

            return {
                "acc": float(avg_r),
                "sample_cnt": int(sample_cnt),
                "trunc": float(sample_trunc / max(1, sample_cnt)),
                "all0_frac": float(all0_cnt / max(1, valid_group_cnt)),
                "zero_var_ratio": float(zero_var_groups / max(1, valid_group_cnt)),
                "adv_nz_ratio": float(adv_nonzero_trajs / max(1, adv_total_trajs)),
                "skipped": False,
            }

        self._log(
            f"GRPO train begin: steps={total_steps} batch_prompts={self.cfg.batch_prompts} "
            f"group={self.cfg.group_size} ppo_epochs={self.cfg.ppo_epochs} "
            f"lr={self.cfg.lr} ratio_clip={self.cfg.clip_range} ent={self.cfg.entropy_coef}"
        )
        if not self.cfg.memory_efficient_adamw:
            self._log(f"Optimizer: Adam (no weight_decay), eps={self.cfg.optimizer_eps}")
        st0 = self._model_stats()
        self._log(f"model init: absmax={st0['absmax']:.6f} rms={st0['rms_avg']:.6f} bad={st0['bad']}")
        p0 = next(self.model.parameters())
        self._log(f"model dtype={p0.dtype}, device={p0.device}  (expect float32 for fp32 training)")

        do_initial_sync = bool((self.cfg.eval_before_train or self.cfg.overfit_test) and self.infer is not None)
        if do_initial_sync:
            self._sync_infer_weights(step=0, force=True)
        if self.cfg.eval_before_train and self.infer is not None:
            if self.is_main:
                self._log("[EVAL] running step 0 eval before training...")
            self.model.eval()
            self.evaluate(step=0)
        self._dist_barrier()

        if bool(self.cfg.overfit_test):
            overfit_batch_n = min(max(1, int(self.cfg.overfit_batch_n)), len(self.data))
            if overfit_batch_n <= 0:
                raise RuntimeError("overfit_test failed: empty training data.")
            overfit_max_rounds = max(1, int(getattr(self.cfg, "overfit_max_rounds", 20)))
            chosen_seed = int(self.seed)
            if len(self.data) <= overfit_batch_n:
                overfit_batch = list(self.data[:overfit_batch_n])
            else:
                rr = random.Random(chosen_seed)
                idxs = rr.sample(range(len(self.data)), overfit_batch_n)
                overfit_batch = [self.data[i] for i in idxs]
            self._overfit_rollout_seed = int(chosen_seed)
            self._probe_rollout_seed = int(chosen_seed)
            self._log(
                f"[OVERFIT] selected fixed batch directly: questions={overfit_batch_n} seed={chosen_seed}"
            )
            if self.world_size > 1:
                local_q_counts = [len(overfit_batch[r::self.world_size]) for r in range(int(self.world_size))]
                self._log(
                    f"[OVERFIT] distributed fixed-batch mode: sharded global_questions={overfit_batch_n} "
                    f"local_questions={local_q_counts}"
                )
            rec_baseline = _run_one_step(
                step=0,
                total_steps_local=overfit_max_rounds,
                phase="overfit_baseline_probe",
                fixed_batch=overfit_batch,
                run_periodic_hooks=False,
                update_policy=False,
                record_outputs=True,
                force_sync=True,
            )
            baseline_acc = float(rec_baseline.get("acc", 0.0))
            self._log(
                f"[OVERFIT] baseline_for_compare acc={baseline_acc:.3f} "
                f"trunc={float(rec_baseline.get('trunc', 1.0)):.3f}"
            )
            if bool(getattr(self.cfg, "overfit_probe_only", False)):
                self._log("[OVERFIT] probe-only mode enabled; skip train rounds.")
                return

            consec_hit = 0
            passed = False
            use_perfect_acc_gate = (overfit_batch_n == 1)
            use_high_acc_gate = (baseline_acc >= 0.95)
            target_gain = 0.2
            def _overfit_rollout_stop_reason(acc_now: float, prev_consec: int) -> Optional[str]:
                if use_perfect_acc_gate:
                    next_consec = prev_consec + 1 if acc_now >= (1.0 - 1e-12) else 0
                    if next_consec >= 4:
                        return f"rollout already reaches target_acc=1.000 with consec_hit={next_consec}/4"
                    return None
                if use_high_acc_gate:
                    next_consec = prev_consec + 1 if acc_now >= (baseline_acc - 1e-12) else 0
                    if next_consec >= 4:
                        return f"rollout already maintains target_acc>={baseline_acc:.3f} with consec_hit={next_consec}/4"
                    return None
                next_consec = prev_consec + 1 if (acc_now - baseline_acc) >= target_gain else 0
                if next_consec >= 4:
                    return f"rollout already reaches target_gain={target_gain:.3f} with consec_hit={next_consec}/4"
                return None

            for r in range(1, overfit_max_rounds + 1):
                rec = _run_one_step(
                    step=r,
                    total_steps_local=overfit_max_rounds,
                    phase="overfit",
                    fixed_batch=overfit_batch,
                    run_periodic_hooks=False,
                    rollout_stop_check=(lambda acc_now, prev_consec=consec_hit: _overfit_rollout_stop_reason(acc_now, prev_consec)),
                )
                acc_now = float(rec.get("acc", 0.0))
                gain = float(acc_now - baseline_acc)
                if use_perfect_acc_gate:
                    if acc_now >= (1.0 - 1e-12):
                        consec_hit += 1
                    else:
                        consec_hit = 0
                    self._log(
                        f"[OVERFIT] round={r} acc={acc_now:.3f} baseline={baseline_acc:.3f} "
                        f"target_acc=1.000 consec_hit={consec_hit}/4 (seed={chosen_seed})"
                    )
                elif use_high_acc_gate:
                    if acc_now >= (baseline_acc - 1e-12):
                        consec_hit += 1
                    else:
                        consec_hit = 0
                    self._log(
                        f"[OVERFIT] round={r} acc={acc_now:.3f} baseline={baseline_acc:.3f} "
                        f"target_acc>={baseline_acc:.3f} consec_hit={consec_hit}/4 (seed={chosen_seed})"
                    )
                else:
                    if gain >= target_gain:
                        consec_hit += 1
                    else:
                        consec_hit = 0
                    self._log(
                        f"[OVERFIT] round={r} acc={acc_now:.3f} baseline={baseline_acc:.3f} "
                        f"gain={gain:.3f} target_gain={target_gain:.3f} consec_hit={consec_hit}/4 (seed={chosen_seed})"
                    )
                if consec_hit >= 4:
                    passed = True
                    if self.infer is not None:
                        self._log(f"[OVERFIT] pass on rollout at round={r}; running eval before formal training.")
                        self.model.eval()
                        self.evaluate(step=r)
                    break
            if not passed:
                if use_perfect_acc_gate:
                    raise RuntimeError(
                        f"overfit_test failed: seed={chosen_seed}, baseline={baseline_acc:.3f}, "
                        f"did not achieve 4 consecutive rounds with acc >= 1.000 within {overfit_max_rounds} rounds."
                    )
                if use_high_acc_gate:
                    raise RuntimeError(
                        f"overfit_test failed: seed={chosen_seed}, baseline={baseline_acc:.3f}, "
                        f"did not maintain acc >= {baseline_acc:.3f} for 4 consecutive rounds within {overfit_max_rounds} rounds."
                    )
                raise RuntimeError(
                    f"overfit_test failed: seed={chosen_seed}, baseline={baseline_acc:.3f}, "
                    f"did not achieve 4 consecutive rounds with acc gain >= {target_gain:.3f} within {overfit_max_rounds} rounds."
                )
            self._log("[OVERFIT] pass, start formal training.")
            self._dist_barrier()

        for step in range(1, total_steps + 1):
            _run_one_step(
                step=step,
                total_steps_local=total_steps,
                phase="train",
                fixed_batch=None,
                run_periodic_hooks=True,
            )

        if self.cfg.enable_faulthandler and float(self.cfg.hang_dump_s) > 0:
            try:
                import faulthandler
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass

        self._log("train end.")


# =========================================================
# Sanity check: train vs infer last-logits top1
# =========================================================

@torch.no_grad()
def sanity_check_train_vs_fp16(train_model, infer_model, infer_engine, encode, decode, train_device="cuda:0", infer_device="cuda:1"):
    prompt = "User: What is 2+2? think\nAssistant: <think>\n"
    ids = encode(prompt)
    if not ids:
        raise RuntimeError("encode(prompt) returned empty")

    # Train model forward on train_device
    t = torch.tensor([ids], device=train_device, dtype=torch.long)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits_train = train_model(t)[0, -1].float().cpu()

    # Infer model forward on infer_device
    infer_idx = _parse_cuda_index(infer_device)
    if infer_idx is not None:
        torch.cuda.set_device(infer_idx)
    state = infer_engine.init_state_with_time_state(B=1)
    out2 = infer_model.forward_batch([ids], state)
    if torch.is_tensor(out2) and out2.dim() == 3:
        out2 = out2[:, -1, :]
    logits_infer = out2[0].float().cpu()

    top_train = int(torch.argmax(logits_train).item())
    top_infer = int(torch.argmax(logits_infer).item())

    print("[sanity] top1_train =", top_train, "->", repr(decode([top_train])), flush=True)
    print("[sanity] top1_infer =", top_infer, "->", repr(decode([top_infer])), flush=True)

    if top_train != top_infer:
        # For dual-GPU mode, allow mismatch as warning due to fp16/bf16 numerical differences
        print("[sanity] WARNING: train vs infer top1 mismatch - may be due to fp16/bf16 precision diff", flush=True)
        # Check if top5 overlaps
        top5_train = torch.topk(logits_train, 5).indices.tolist()
        top5_infer = torch.topk(logits_infer, 5).indices.tolist()
        overlap = len(set(top5_train) & set(top5_infer))
        print(f"[sanity] top5_train = {top5_train}", flush=True)
        print(f"[sanity] top5_infer = {top5_infer}", flush=True)
        print(f"[sanity] top5 overlap = {overlap}/5", flush=True)
        if overlap < 3:
            raise RuntimeError("SANITY FAIL: train vs infer top5 has <3 overlap - check model loading")


def _init_distributed_from_env() -> Tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return 0, 1, 0
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE>1 was requested.")
    if not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 requires CUDA, but CUDA is unavailable.")
    n_cuda = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= n_cuda:
        raise RuntimeError(f"LOCAL_RANK={local_rank} out of range for visible CUDA devices={n_cuda}.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count()))))
    print(f"[dist] ready rank={rank} local_rank={local_rank} world_size={world_size} cuda={torch.cuda.current_device()}", flush=True)
    return rank, world_size, local_rank


# =========================================================
# Main
# =========================================================

def main():
    cfg_defaults = GRPOConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="rwkv7-g1d-2.9b-20260131-ctx8192", help="model path")
    ap.add_argument("--train_jsonl", type=str, default="train.jsonl")
    ap.add_argument("--eval_data", type=str, default="math192.jsonl", help="eval dataset path")
    ap.add_argument("--out_dir", type=str, default=None, help="output directory (default: out_grpo_<timestamp>)")

    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1, help="Gradient checkpointing: 0=off, 1=on (saves memory)")

    ap.add_argument("--batch_prompts", type=int, default=cfg_defaults.batch_prompts)
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--max_rollout_rounds", type=int, default=cfg_defaults.max_rollout_rounds, help="rollout rounds per training step")
    ap.add_argument("--rollout_update_interval", type=int, default=cfg_defaults.rollout_update_interval, help="rollout policy update interval (steps)")
    ap.add_argument("--no_rollout_cache", action="store_true", help="disable rollout cache and use live model")
    ap.add_argument("--rollout_ema_decay", type=float, default=cfg_defaults.rollout_ema_decay, help="EMA decay for rollout policy (0 to disable)")
    ap.add_argument("--sync_infer_interval", type=int, default=cfg_defaults.sync_infer_interval, help="sync train->rollout policy every N steps (reference backend)")
    ap.add_argument("--sync_infer_offload_cpu", action="store_true", help="offload FSDP full-params to CPU during train->rollout sync")
    ap.add_argument("--no_sync_infer_offload_cpu", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=cfg_defaults.top_k)
    ap.add_argument("--decay", type=float, default=cfg_defaults.decay)
    ap.add_argument("--mask_token0", action="store_true")
    ap.add_argument(
        "--rollout_backend",
        type=str,
        default="reference",
        choices=["reference", "train"],
        help="reference: fp16 incremental rollout model (fast). train: training-model rollout (debug, much slower).",
    )

    # Rapid-Sampling repetition penalty
    ap.add_argument("--use_rapid_sampling", action="store_true", help="enable Rapid-Sampling kernel")
    ap.add_argument("--no_rapid_sampling", action="store_true", help="disable Rapid-Sampling kernel")
    ap.add_argument("--presence_penalty", type=float, default=cfg_defaults.presence_penalty)
    ap.add_argument("--repetition_penalty", type=float, default=cfg_defaults.repetition_penalty)
    ap.add_argument("--penalty_decay", type=float, default=cfg_defaults.penalty_decay)

    ap.add_argument("--ppo_epochs", type=int, default=cfg_defaults.ppo_epochs)
    ap.add_argument("--lr", type=float, default=cfg_defaults.lr)
    ap.add_argument("--beta1", type=float, default=cfg_defaults.beta1)
    ap.add_argument("--beta2", type=float, default=cfg_defaults.beta2)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--format_reward_coef", type=float, default=cfg_defaults.format_reward_coef, help="coefficient for RWKV format reward added on top of answer reward")
    ap.add_argument("--online_correct_only_ce", dest="online_correct_only_ce", action="store_true", help="ignore wrong trajectories and do CE only on sampled correct trajectories")
    ap.add_argument("--no_online_correct_only_ce", dest="online_correct_only_ce", action="store_false", help="disable correct-only CE and use the PPO/GRPO branch")
    ap.add_argument("--clip_range", type=float, default=cfg_defaults.clip_range)
    ap.add_argument("--grad_clip", type=float, default=cfg_defaults.grad_clip)
    ap.add_argument("--ppo_max_token_len_per_gpu", type=int, default=cfg_defaults.ppo_max_token_len_per_gpu)
    ap.add_argument("--diag_inner_update", action="store_true", help="diagnostic: log teacher-forced stats after every PPO inner epoch")
    ap.add_argument("--diag_zero_padding", action="store_true", help="diagnostic: compare grads before/after zero-padding backward")
    ap.add_argument("--diag_compare_global_grad", action="store_true", help="diagnostic: compare distributed accumulated grad with rank0 oracle grad on the same merged trajectories")
    ap.add_argument("--diag_compare_step", type=int, default=cfg_defaults.diag_compare_step)
    ap.add_argument("--diag_compare_epoch", type=int, default=cfg_defaults.diag_compare_epoch)
    ap.add_argument("--diag_compare_param_count", type=int, default=cfg_defaults.diag_compare_param_count)
    ap.add_argument("--use_ppo_loss", action="store_true", default=True, help="use PPO clipped ratio objective")
    ap.add_argument("--no_ppo_loss", action="store_true", help="disable PPO ratio objective and use Qwen-style vanilla PG")
    ap.add_argument("--advantage_eps", type=float, default=cfg_defaults.advantage_eps, help="epsilon in group reward normalization")
    ap.add_argument("--optimizer_eps", type=float, default=cfg_defaults.optimizer_eps)
    ap.add_argument("--memory_efficient_adamw", action="store_true", help="offload optimizer states to CPU pinned memory")
    ap.add_argument("--no_memory_efficient_adamw", action="store_true")
    ap.add_argument("--qwen_align_common", action="store_true", help="align generic training logic/reward handling with GRPO-Zero Qwen")

    ap.add_argument("--log_interval", type=int, default=cfg_defaults.log_interval)
    ap.add_argument("--save_interval", type=int, default=cfg_defaults.save_interval)
    ap.add_argument("--infer_check_interval", type=int, default=cfg_defaults.infer_check_interval)

    ap.add_argument("--eval_interval", type=int, default=cfg_defaults.eval_interval)
    ap.add_argument("--eval_n", type=int, default=cfg_defaults.eval_n)
    ap.add_argument("--eval_temperature", type=float, default=cfg_defaults.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=cfg_defaults.eval_top_p)
    ap.add_argument("--eval_top_k", type=int, default=cfg_defaults.eval_top_k)
    ap.add_argument("--eval_max_new_tokens", type=int, default=cfg_defaults.eval_max_new_tokens)
    ap.add_argument("--eval_presence_penalty", type=float, default=cfg_defaults.eval_presence_penalty)
    ap.add_argument("--eval_frequency_penalty", type=float, default=cfg_defaults.eval_frequency_penalty)
    ap.add_argument("--eval_penalty_decay", type=float, default=cfg_defaults.eval_penalty_decay)
    ap.add_argument("--eval_before_train", action="store_true", help="run eval at step 0 before training")
    ap.add_argument("--overfit_test", action="store_true", help="run pre-train fixed-batch overfit sanity test")
    ap.add_argument("--overfit_batch_n", type=int, default=cfg_defaults.overfit_batch_n, help="number of questions used by overfit_test")
    ap.add_argument("--overfit_max_rounds", type=int, default=cfg_defaults.overfit_max_rounds, help="max overfit rounds used by overfit_test")
    ap.add_argument("--overfit_probe_only", action="store_true", help="run overfit baseline probe only, then exit")
    ap.add_argument("--model_init", type=str, default=None, help="optional: load full-parameter init checkpoint")
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument(
        "--sanity_check_infer",
        action="store_true",
        help="run train-vs-reference-infer sanity check when rollout_backend=reference",
    )

    ap.add_argument("--enable_faulthandler", action="store_true")
    ap.add_argument("--hang_dump_s", type=float, default=cfg_defaults.hang_dump_s)

    ap.set_defaults(
        use_rapid_sampling=True,
        online_correct_only_ce=True,
        memory_efficient_adamw=True,
        sync_infer_offload_cpu=True,
    )
    args = ap.parse_args()
    if args.no_rapid_sampling:
        args.use_rapid_sampling = False

    if args.no_ppo_loss:
        args.use_ppo_loss = False

    if args.no_memory_efficient_adamw:
        args.memory_efficient_adamw = False

    if args.no_sync_infer_offload_cpu:
        args.sync_infer_offload_cpu = False

    if args.qwen_align_common:
        args.use_ppo_loss = False
        args.entropy_coef = 0.0
        args.beta2 = 0.999
        args.advantage_eps = 1e-4
        args.optimizer_eps = 1e-8
        if not args.memory_efficient_adamw:
            args.memory_efficient_adamw = False

    rank, world_size, local_rank = _init_distributed_from_env()
    is_main = (rank == 0)

    def rank0_print(msg: str):
        if is_main:
            print(msg, flush=True)

    auto_out_dir = False
    if args.out_dir is None or str(args.out_dir).strip() == "":
        auto_out_dir = True
        if is_main:
            args.out_dir = f"out_grpo_{now_str()}"
            rank0_print(f"[AUTO-OUT] using out_dir={args.out_dir}")
    if world_size > 1 and _dist_is_initialized() and auto_out_dir:
        out_obj = [args.out_dir if is_main else None]
        dist.broadcast_object_list(out_obj, src=0)
        args.out_dir = str(out_obj[0])

    judge_dir = os.path.join(args.out_dir, "judge_pipeline")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")
    n_cuda = torch.cuda.device_count()
    if n_cuda < 1:
        raise RuntimeError("No visible CUDA devices.")

    train_gpu_indices_all = list(range(n_cuda))
    if world_size > len(train_gpu_indices_all):
        raise RuntimeError(
            f"WORLD_SIZE={world_size} exceeds available training GPUs={len(train_gpu_indices_all)} "
            f"(train_gpus={train_gpu_indices_all})."
        )

    train_gpu_indices = train_gpu_indices_all[:max(1, world_size)]
    if world_size > 1:
        if local_rank >= world_size:
            raise RuntimeError(f"LOCAL_RANK={local_rank} out of range for WORLD_SIZE={world_size}.")
        # Keep LOCAL_RANK -> cuda:LOCAL_RANK mapping for NCCL/FSDP stability.
        train_gpu = int(local_rank)
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        rank0_print(
            f"[GPU] distributed mode: world_size={world_size} "
            f"train_gpus={train_gpu_indices} (rank{rank} -> cuda:{train_gpu})"
        )
    else:
        train_gpu = train_gpu_indices[0]
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        if len(train_gpu_indices_all) > 1:
            rank0_print(
                f"[GPU] WARN: single-process mode uses one training GPU only ({device}). "
                f"Use torchrun --nproc_per_node={len(train_gpu_indices_all)} to use all train GPUs."
            )
        rank0_print(
            f"[GPU] single-process mode: total={n_cuda}, train_gpu={train_gpu}"
        )

    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
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

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_data(args.train_jsonl)
    raw_train_n = len(data)
    data = filter_train_data_by_problem_len(data, max_problem_len=512)
    filtered_n = raw_train_n - len(data)
    rank0_print(
        f"[DATA] train length filter: max_problem_len=512 kept={len(data)}/{raw_train_n} filtered={filtered_n}"
    )
    if not data:
        raise RuntimeError("empty train data")

    eval_data = None
    if args.eval_data:
        eval_data = load_data(args.eval_data)
        rank0_print(f"Loaded eval data: {len(eval_data)} samples from {args.eval_data}")
    else:
        rank0_print("No eval_data specified, using train data for eval")

    from utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)

    encode = lambda s: tok.encode(s)

    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except TypeError:
            pass
        try:
            return tok.decode(ids)
        except UnicodeDecodeError:
            try:
                b = tok.decodeBytes(ids)
                return b.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(int(x) % 256) for x in ids)

    decode = safe_decode

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    rank0_print(f"Loading model: {pth_path}")
    rank0_print(f"Train GPUs (all): {train_gpu_indices_all}, Active train ranks: {world_size}")

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

    if args.model_init:
        ok = load_model_init(train_model_raw, args.model_init)
        rank0_print(f"[model_init] loaded={ok} from {args.model_init}")

    cfg = GRPOConfig(
        batch_prompts=int(args.batch_prompts),
        group_size=int(args.group_size),
        max_rollout_rounds=int(args.max_rollout_rounds),
        rollout_update_interval=int(args.rollout_update_interval),
        rollout_use_cache=not bool(args.no_rollout_cache),
        rollout_ema_decay=float(args.rollout_ema_decay),
        sync_infer_interval=max(1, int(args.sync_infer_interval)),
        sync_infer_offload_cpu=bool(args.sync_infer_offload_cpu),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        decay=float(args.decay),
        mask_token0=bool(args.mask_token0),
        use_rapid_sampling=bool(args.use_rapid_sampling),
        presence_penalty=float(args.presence_penalty),
        repetition_penalty=float(args.repetition_penalty),
        penalty_decay=float(args.penalty_decay),
        ppo_epochs=int(args.ppo_epochs),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        entropy_coef=float(args.entropy_coef),
        clip_range=float(args.clip_range),
        grad_clip=float(args.grad_clip),
        optimizer_eps=float(args.optimizer_eps),
        ppo_max_token_len_per_gpu=int(args.ppo_max_token_len_per_gpu),
        diag_inner_update=bool(args.diag_inner_update),
        diag_zero_padding=bool(args.diag_zero_padding),
        diag_compare_global_grad=bool(args.diag_compare_global_grad),
        diag_compare_step=int(args.diag_compare_step),
        diag_compare_epoch=int(args.diag_compare_epoch),
        diag_compare_param_count=int(args.diag_compare_param_count),
        use_ppo_loss=bool(args.use_ppo_loss),
        advantage_eps=float(args.advantage_eps),
        memory_efficient_adamw=bool(args.memory_efficient_adamw),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        infer_check_interval=int(args.infer_check_interval),
        eval_interval=int(args.eval_interval),
        eval_n=int(args.eval_n),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        eval_max_new_tokens=int(args.eval_max_new_tokens),
        eval_presence_penalty=float(args.eval_presence_penalty),
        eval_frequency_penalty=float(args.eval_frequency_penalty),
        eval_penalty_decay=float(args.eval_penalty_decay),
        eval_before_train=bool(args.eval_before_train),
        overfit_test=bool(args.overfit_test),
        overfit_batch_n=int(args.overfit_batch_n),
        overfit_max_rounds=max(1, int(args.overfit_max_rounds)),
        overfit_probe_only=bool(args.overfit_probe_only),
        format_reward_coef=float(args.format_reward_coef),
        online_correct_only_ce=bool(args.online_correct_only_ce),
        enable_faulthandler=bool(args.enable_faulthandler),
        hang_dump_s=float(args.hang_dump_s),
    )

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
            rank0_print("[GPU] FSDP mixed_precision disabled by default (RWKV+FSDP bf16 grad mismatch); forward autocast remains enabled")
        auto_wrap_policy = None
        if hasattr(train_model_raw, "blocks") and len(getattr(train_model_raw, "blocks", [])) > 0:
            block_cls = type(train_model_raw.blocks[0])
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={block_cls},
            )
        train_model = FSDP(
            train_model_raw,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
            mixed_precision=fsdp_mp,
            auto_wrap_policy=auto_wrap_policy,
        )
        rank0_print(f"[GPU] FSDP enabled across {world_size} training ranks.")

    def judge_log(msg: str):
        if not is_main:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{ts}] {msg}", flush=True)

    judge_client = FileJudgeClient(
        judge_dir=judge_dir,
        timeout_s=3600.0,
        poll_interval_s=2.0,
        log_fn=judge_log,
    )

    if args.rollout_backend == "reference":
        if world_size > 1 and cfg.rollout_use_cache:
            cfg.rollout_use_cache = False
            rank0_print("[rollout] disable rollout cache in distributed reference backend (avoid stale local-state snapshot).")
        infer_model, _ = load_infer_model_fp16(base_name, device=device)
        infer_engine = FP16BatchInference(
            infer_model,
            train_model_raw,
            encode,
            decode,
            device=device,
            cfg=cfg,
        )
        rank0_print(f"[GPU] rollout engine: reference fp16 on {device}")
        if args.sanity_check_infer and is_main:
            sanity_check_train_vs_fp16(
                train_model_raw,
                infer_model,
                infer_engine,
                encode,
                decode,
                train_device=device,
                infer_device=device,
            )
    else:
        infer_engine = TrainModelRolloutEngine(
            model=train_model,
            decode_fn=decode,
            device=device,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
        )
        rank0_print("[GPU] rollout engine: training model itself (debug backend)")
        if args.sanity_check_infer and is_main:
            rank0_print("[sanity] skipped: no standalone infer model in train rollout backend.")

    trainer = GRPOFullFinetuneTrainer(
        train_model=train_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        data=data,
        judge_client=judge_client,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
        eval_data=eval_data,
        train_gpu_count=(world_size if world_size > 1 else 1),
        rank=rank,
        world_size=world_size,
        model_pth_path=pth_path,
        model_init_path=str(args.model_init or ""),
    )

    try:
        trainer.train(total_steps=int(args.total_steps))
        if world_size > 1 and _dist_is_initialized():
            torch.cuda.set_device(train_gpu)
            dist.barrier()
    finally:
        if world_size > 1 and _dist_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
