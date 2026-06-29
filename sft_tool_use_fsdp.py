#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import functools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from grpo_math_local import (
    FSDP,
    FullStateDictConfig,
    MemoryEfficientAdamW,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    _dist_is_initialized,
    _ensure_cuda_toolkit_env,
    _init_distributed_from_env,
    _parse_cuda_index,
    _resolve_torch_cuda_arch_list,
    build_prompt,
    enable_full_finetune,
    load_data,
    load_train_model_rwkv7_cuda,
    normalize_model_arg,
    now_str,
    transformer_auto_wrap_policy,
)


TOOL_CALL_TEMPLATE = "<tool_call>{tool_name}</tool_call>\n"


@dataclass
class SFTConfig:
    train_jsonl: str
    out_dir: str
    model: str
    tokenizer: str
    max_steps: int
    ctx_len: int
    grad_cp: int
    seed: int
    lr: float
    beta1: float
    beta2: float
    optimizer_eps: float
    grad_clip: float
    save_interval: int
    log_interval: int
    micro_batch_size: int
    global_batch_size: int
    memory_efficient_adamw: bool


def _append_text_line(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _ensure_trailing_newline(text: str) -> str:
    text = str(text or "")
    if not text.endswith("\n"):
        text += "\n"
    return text


def _tool_name_from_record(rec: Dict[str, Any]) -> str:
    tools = rec.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function")
            if isinstance(func, dict):
                name = str(func.get("name", "")).strip()
                if name:
                    return name
    return "tool"


def _format_tool_call(name: str, arguments_raw: Any) -> str:
    tool_name = str(name or "tool").strip() or "tool"
    parsed_args = None
    if isinstance(arguments_raw, str):
        raw_text = arguments_raw.strip()
        if raw_text:
            try:
                parsed_args = json.loads(raw_text)
            except Exception:
                parsed_args = None
    elif isinstance(arguments_raw, dict):
        parsed_args = arguments_raw

    if isinstance(parsed_args, dict) and isinstance(parsed_args.get("code"), str):
        code = parsed_args.get("code", "").rstrip()
        return (
            f'<tool_call name="{tool_name}">\n'
            "```python\n"
            f"{code}\n"
            "```\n"
            "</tool_call>\n"
        )

    if parsed_args is not None:
        args_text = json.dumps(parsed_args, ensure_ascii=False, indent=2).rstrip()
    else:
        args_text = str(arguments_raw or "").rstrip()
    return (
        f'<tool_call name="{tool_name}">\n'
        "<arguments>\n"
        f"{args_text}\n"
        "</arguments>\n"
        "</tool_call>\n"
    )


def _render_assistant_message(msg: Dict[str, Any]) -> Tuple[str, bool]:
    parts: List[str] = []
    reasoning_content = str(msg.get("reasoning_content") or "")
    if reasoning_content.strip():
        parts.append(_ensure_trailing_newline(reasoning_content.strip()))

    content = str(msg.get("content") or "")
    if content.strip():
        parts.append(_ensure_trailing_newline(content.strip()))

    tool_calls = msg.get("tool_calls") or []
    has_structured_tool_call = False
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if not isinstance(func, dict):
                continue
            parts.append(_format_tool_call(func.get("name"), func.get("arguments")))
            has_structured_tool_call = True
    return "".join(parts), has_structured_tool_call


def _add_segment(tokens: List[int], loss_mask: List[int], text: str, supervise: bool, encode_fn):
    if not text:
        return
    ids = encode_fn(text)
    if not ids:
        return
    tokens.extend(int(x) for x in ids)
    loss_mask.extend([1 if supervise else 0] * len(ids))


def build_sft_example(
    rec: Dict[str, Any],
    encode_fn,
    ctx_len: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    problem = str(rec.get("problem", "")).strip()
    if not problem:
        messages = rec.get("messages") or []
        for msg in messages:
            if str(msg.get("role", "")).strip() == "user":
                problem = str(msg.get("content", "")).strip()
                if problem:
                    break
    if not problem:
        return None, "missing_problem"

    messages = rec.get("messages") or []
    tool_name = _tool_name_from_record(rec)
    tool_call_text = TOOL_CALL_TEMPLATE.format(tool_name=tool_name)

    tokens: List[int] = []
    loss_mask: List[int] = []
    _add_segment(tokens, loss_mask, build_prompt(problem), False, encode_fn)

    needs_assistant_prefix = False
    skipped_first_user = False
    assistant_turns = 0
    pending_structured_tool_call = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "") or "")
        if role == "system":
            continue
        if role == "user":
            if not skipped_first_user:
                skipped_first_user = True
                continue
            _add_segment(tokens, loss_mask, f"User: {content.strip()}\n", False, encode_fn)
            needs_assistant_prefix = True
            pending_structured_tool_call = False
            continue
        if role == "assistant":
            rendered_content, has_structured_tool_call = _render_assistant_message(msg)
            if not rendered_content:
                pending_structured_tool_call = False
                continue
            if needs_assistant_prefix:
                _add_segment(tokens, loss_mask, "Assistant: ", False, encode_fn)
            _add_segment(tokens, loss_mask, _ensure_trailing_newline(rendered_content), True, encode_fn)
            needs_assistant_prefix = True
            assistant_turns += 1
            pending_structured_tool_call = has_structured_tool_call
            continue
        if role == "tool":
            if not pending_structured_tool_call and needs_assistant_prefix:
                _add_segment(tokens, loss_mask, "Assistant: ", False, encode_fn)
                _add_segment(tokens, loss_mask, tool_call_text, True, encode_fn)
            _add_segment(tokens, loss_mask, f"Tool: {content.rstrip()}\n", False, encode_fn)
            needs_assistant_prefix = True
            pending_structured_tool_call = False
            continue

    if assistant_turns <= 0:
        return None, "missing_assistant"
    if len(tokens) < 2:
        return None, "too_short"
    target_tokens = int(sum(loss_mask[1:]))
    if target_tokens <= 0:
        return None, "missing_target"
    if len(tokens) > int(ctx_len):
        return None, "too_long"
    return {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "target_tokens": target_tokens,
        "total_tokens": len(tokens),
        "uuid": str(rec.get("uuid", "")),
        "problem": problem,
        "tool_name": tool_name,
    }, "ok"


def preprocess_dataset(
    records: List[Dict[str, Any]],
    encode_fn,
    ctx_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    lengths: List[int] = []
    target_lengths: List[int] = []
    first_preview: Optional[Dict[str, Any]] = None

    for rec in records:
        ex, reason = build_sft_example(
            rec,
            encode_fn=encode_fn,
            ctx_len=ctx_len,
        )
        if ex is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        built.append(ex)
        lengths.append(int(ex["total_tokens"]))
        target_lengths.append(int(ex["target_tokens"]))
        if first_preview is None:
            first_preview = {
                "uuid": ex["uuid"],
                "problem_preview": ex["problem"][:200],
                "tool_name": ex["tool_name"],
                "total_tokens": ex["total_tokens"],
                "target_tokens": ex["target_tokens"],
            }

    stats = {
        "loaded_records": len(records),
        "valid_records": len(built),
        "skipped": skipped,
        "total_tokens_mean": (sum(lengths) / max(1, len(lengths))) if lengths else 0.0,
        "total_tokens_max": max(lengths) if lengths else 0,
        "target_tokens_mean": (sum(target_lengths) / max(1, len(target_lengths))) if target_lengths else 0.0,
        "target_tokens_max": max(target_lengths) if target_lengths else 0,
        "first_preview": first_preview,
    }
    return built, stats


def _all_reduce_scalar(value: float, device: str, op=dist.ReduceOp.SUM) -> float:
    if not _dist_is_initialized():
        return float(value)
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return float(t.item())


def _collect_model_state(model: torch.nn.Module, base_model: torch.nn.Module, fsdp_enabled: bool) -> Optional[Dict[str, torch.Tensor]]:
    if fsdp_enabled:
        if FSDP is None or StateDictType is None or FullStateDictConfig is None:
            raise RuntimeError("FSDP state-dict helpers are unavailable in current torch build.")
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            state = model.state_dict()
        if dist.is_initialized() and dist.get_rank() != 0:
            return None
        return state
    return {n: p.detach().cpu() for n, p in base_model.named_parameters()}


class ShardedIterator:
    def __init__(self, dataset: List[Dict[str, Any]], rank: int, world_size: int, seed: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self._sampler = None
        self._iter = None
        if world_size > 1:
            self._sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
                seed=seed,
            )
        self._reset_iter()

    def _reset_iter(self):
        if self.world_size > 1:
            self._sampler.set_epoch(self.epoch)
            self._iter = iter(list(self._sampler))
        else:
            idxs = list(range(len(self.dataset)))
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(idxs)
            self._iter = iter(idxs)
        self.epoch += 1

    def next(self) -> Dict[str, Any]:
        while True:
            try:
                idx = next(self._iter)
                return self.dataset[int(idx)]
            except StopIteration:
                self._reset_iter()


class Logger:
    def __init__(self, out_dir: str, rank: int, is_main: bool):
        self.out_dir = out_dir
        self.rank = rank
        self.is_main = is_main
        self.log_path = os.path.join(out_dir, "train.log")
        self.rank_log_path = os.path.join(out_dir, f"train_rank{rank}.log")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        if self.is_main:
            print(line, flush=True)
        _append_text_line(self.rank_log_path, line)
        if self.is_main:
            _append_text_line(self.log_path, line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/rwkv7-g1c-2.9b-20251231-ctx8192.pth")
    ap.add_argument("--train_jsonl", type=str, default="tool_sft_train.jsonl")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--optimizer_eps", type=float, default=1e-18)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_interval", type=int, default=20)
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--global_batch_size", type=int, default=16)
    ap.add_argument("--memory_efficient_adamw", action=argparse.BooleanOptionalAction, default=True)
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
        raise RuntimeError("This SFT entry currently expects --micro_batch_size=1 as requested.")

    rank, world_size, local_rank = _init_distributed_from_env()
    is_main = rank == 0

    def rank0_print(msg: str):
        if is_main:
            print(msg, flush=True)

    auto_out_dir = False
    if args.out_dir is None or str(args.out_dir).strip() == "":
        auto_out_dir = True
        if is_main:
            args.out_dir = f"out_sft_tool_use_{now_str()}"
    if world_size > 1 and _dist_is_initialized() and auto_out_dir:
        out_obj = [args.out_dir if is_main else None]
        dist.broadcast_object_list(out_obj, src=0)
        args.out_dir = str(out_obj[0])

    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(args.out_dir, rank=rank, is_main=is_main)

    random.seed(args.seed)
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

    active_train_gpus = available_gpu_indices[:max(1, world_size)]
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
        if len(available_gpu_indices) > 1:
            rank0_print(
                f"[GPU] WARN: single-process mode uses one training GPU only ({device}). "
                f"Use torchrun --nproc_per_node={len(available_gpu_indices)} to use all train GPUs."
            )
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

    sft_cfg = SFTConfig(
        train_jsonl=str(args.train_jsonl),
        out_dir=str(args.out_dir),
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        max_steps=int(args.max_steps),
        ctx_len=int(args.ctx_len),
        grad_cp=int(args.grad_cp),
        seed=int(args.seed),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        optimizer_eps=float(args.optimizer_eps),
        grad_clip=float(args.grad_clip),
        save_interval=int(args.save_interval),
        log_interval=int(args.log_interval),
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
        memory_efficient_adamw=bool(args.memory_efficient_adamw),
    )
    if sft_cfg.global_batch_size <= 0:
        raise RuntimeError("--global_batch_size must be > 0")
    if sft_cfg.global_batch_size % max(1, world_size) != 0:
        raise RuntimeError(
            f"--global_batch_size={sft_cfg.global_batch_size} must be divisible by world_size={world_size}."
        )
    local_batch = sft_cfg.global_batch_size // max(1, world_size)
    if local_batch % sft_cfg.micro_batch_size != 0:
        raise RuntimeError(
            f"Per-rank batch={local_batch} must be divisible by micro_batch_size={sft_cfg.micro_batch_size}."
        )
    accum_steps = local_batch // sft_cfg.micro_batch_size
    if accum_steps <= 0:
        raise RuntimeError("accum_steps resolved to <= 0")

    if is_main:
        with open(os.path.join(args.out_dir, "sft_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(sft_cfg), f, ensure_ascii=False, indent=2)

    data = load_data(args.train_jsonl)
    if not data:
        raise RuntimeError("empty train data")

    from utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)

    sft_data, data_stats = preprocess_dataset(
        data,
        encode_fn=encode,
        ctx_len=int(args.ctx_len),
    )
    if not sft_data:
        raise RuntimeError("No valid SFT sequences after preprocessing.")
    rank0_print(
        "[DATA] preprocessed "
        f"valid={data_stats['valid_records']}/{data_stats['loaded_records']} "
        f"skip={json.dumps(data_stats['skipped'], ensure_ascii=False)} "
        f"tokens(mean/max)={data_stats['total_tokens_mean']:.1f}/{data_stats['total_tokens_max']} "
        f"target(mean/max)={data_stats['target_tokens_mean']:.1f}/{data_stats['target_tokens_max']}"
    )
    if is_main:
        with open(os.path.join(args.out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
            json.dump(data_stats, f, ensure_ascii=False, indent=2)

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")
    rank0_print(f"Loading model: {pth_path}")
    rank0_print(f"Train GPUs (all): {available_gpu_indices}, Active train ranks: {world_size}")

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
            rank0_print("[GPU] FSDP mixed_precision disabled by default (match grpo_math_local.py)")
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

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable params found after wrapping.")

    if args.memory_efficient_adamw:
        opt = MemoryEfficientAdamW(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
            enabled=True,
        )
        logger.log("Optimizer: MemoryEfficientAdamW (CPU-offloaded states)")
    else:
        opt = torch.optim.Adam(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
        )
        logger.log("Optimizer: Adam (no weight_decay)")

    fsdp_enabled = (FSDP is not None) and isinstance(train_model, FSDP)
    iterator = ShardedIterator(sft_data, rank=rank, world_size=world_size, seed=int(args.seed))

    logger.log(
        f"SFT train begin: steps={args.max_steps} global_batch={args.global_batch_size} "
        f"micro_batch={args.micro_batch_size} accum_steps={accum_steps} lr={args.lr:.2e}"
    )
    logger.log(f"FSDP grad accumulation: fsdp_no_sync={bool(args.fsdp_no_sync)}")
    logger.log(
        "SFT supervision: loss is applied only on assistant text "
        "(reasoning_content, assistant content, and assistant tool-call text); "
        "User and Tool segments are masked out."
    )
    logger.log(
        "SFT tool format: assistant reasoning is kept before tool use; structured tool calls "
        "are rendered as `<tool_call name=\"...\">```python ...```</tool_call>`; "
        "Tool outputs are appended as `Tool: ...` context without loss."
    )
    p0 = next(train_model.parameters())
    logger.log(f"model dtype={p0.dtype}, device={p0.device}  (expect float32 for fp32 training)")
    train_model.train()

    try:
        opt.zero_grad(set_to_none=True)
        for step in range(1, int(args.max_steps) + 1):
            step_start = time.time()
            micro_items = [iterator.next() for _ in range(accum_steps)]
            local_target_tokens = int(sum(int(item["target_tokens"]) for item in micro_items))
            global_target_tokens = _all_reduce_scalar(local_target_tokens, device=device, op=dist.ReduceOp.SUM)
            if global_target_tokens <= 0:
                raise RuntimeError(f"Resolved global_target_tokens={global_target_tokens} at step={step}.")

            opt.zero_grad(set_to_none=True)
            step_nll_local = 0.0
            step_tok_local = 0

            for micro_idx, item in enumerate(micro_items):
                sync_now = (micro_idx == (len(micro_items) - 1))
                sync_ctx = contextlib.nullcontext()
                if bool(args.fsdp_no_sync) and fsdp_enabled and hasattr(train_model, "no_sync") and not sync_now:
                    sync_ctx = train_model.no_sync()

                tokens = torch.tensor(item["tokens"], dtype=torch.long, device=device).unsqueeze(0)
                loss_mask = torch.tensor(item["loss_mask"][1:], dtype=torch.float32, device=device).unsqueeze(0)
                inp = tokens[:, :-1].contiguous()
                tgt = tokens[:, 1:].contiguous()

                with sync_ctx:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = train_model(inp)
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)
                    token_nll = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        tgt.reshape(-1),
                        reduction="none",
                    ).reshape_as(tgt)
                    nll_sum = (token_nll * loss_mask).sum()
                    loss = nll_sum / float(global_target_tokens)
                    loss.backward()

                step_nll_local += float(nll_sum.detach().item())
                step_tok_local += int(item["target_tokens"])

            raw_grad_sq_local = 0.0
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        raw_grad_sq_local += float((g.norm(2) ** 2).item())
            raw_grad_sq = _all_reduce_scalar(raw_grad_sq_local, device=device, op=dist.ReduceOp.SUM)
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
            grad_sq = _all_reduce_scalar(grad_sq_local, device=device, op=dist.ReduceOp.SUM)
            grad_norm = math.sqrt(max(0.0, grad_sq))

            opt.step()
            opt.zero_grad(set_to_none=True)

            step_nll = _all_reduce_scalar(step_nll_local, device=device, op=dist.ReduceOp.SUM)
            avg_loss = step_nll / max(1.0, float(global_target_tokens))
            dt = time.time() - step_start

            if is_main and (step == 1 or step == int(args.max_steps) or step % max(1, int(args.log_interval)) == 0):
                logger.log(
                    f"[train step {step}/{int(args.max_steps)}] "
                    f"loss={avg_loss:.6f} target_tok={int(global_target_tokens)} "
                    f"raw_grad={raw_grad:.6f} grad={grad_norm:.6f} "
                    f"lr={float(opt.param_groups[0]['lr']):.2e} step_time={dt:.2f}s"
                )

            if int(args.save_interval) > 0 and (step % int(args.save_interval) == 0 or step == int(args.max_steps)):
                model_state = _collect_model_state(train_model, train_model_raw, fsdp_enabled=fsdp_enabled)
                if is_main and model_state is not None:
                    ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pth")
                    torch.save(
                        {
                            "time": now_str(),
                            "step": step,
                            "cfg": asdict(sft_cfg),
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


if __name__ == "__main__":
    main()
