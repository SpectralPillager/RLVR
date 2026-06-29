#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWKV GSM8K OPSD prototype.

Implements the core idea from arXiv:2601.18734, "Self-Distilled Reasoner:
On-Policy Self-Distillation for Large Language Models":

- student samples on-policy continuations from the normal question prompt;
- teacher is the same RWKV model evaluated under a privileged prompt that
  includes the GSM8K reference solution/answer;
- student is trained on the sampled continuation tokens by minimizing
  teacher/student per-token distribution divergence, with optional token-level
  divergence clipping.

This file is intentionally standalone and does not launch anything by itself.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(REPO_DIR, "reference")
RAPID_SAMPLING_DIR = os.path.join(REPO_DIR, "Rapid-Sampling-main")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, REFERENCE_DIR)

LOCAL_TMP_DIR = os.path.join(REPO_DIR, ".local_tmp")
LOCAL_TORCH_EXTENSIONS_DIR = os.path.join(REPO_DIR, ".torch_extensions")
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_TORCH_EXTENSIONS_DIR, exist_ok=True)
os.environ["TMPDIR"] = LOCAL_TMP_DIR
os.environ["TEMP"] = LOCAL_TMP_DIR
os.environ["TMP"] = LOCAL_TMP_DIR
os.environ["TORCH_EXTENSIONS_DIR"] = LOCAL_TORCH_EXTENSIONS_DIR

import torch
import torch.nn.functional as F

from grpo_math_local单卡 import (
    FP16BatchInference,
    GRPOConfig,
    MemoryEfficientAdamW,
    Muon2DWithAdamWFallback,
    build_prompt,
    extract_gsm8k_final_answer,
    load_data,
    load_infer_model_fp16,
    load_train_model_rwkv7_cuda,
    normalize_model_arg,
    now_str,
    policy_logits_for_logprob,
)
from utils import TRIE_TOKENIZER


@dataclass
class OPSDConfig:
    model: str = "/data_temp/mnt/raid5/zjx/rwkv/rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    teacher_model: str = ""
    train_jsonl: str = "gsmk8ktrain.parquet"
    out_dir: str = ""
    tokenizer: str = "reference/rwkv_vocab_v20230424.txt"
    device: str = "cuda:0"
    ctx_len: int = 8192
    grad_cp: int = 1
    seed: int = 42

    max_steps: int = 300
    batch_prompts: int = 16
    group_size: int = 1
    micro_batch_size: int = 1
    max_new_tokens: int = 512
    rollout_batch_size: int = 128
    update_token_budget: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    use_rapid_sampling: bool = True
    presence_penalty: float = 0.5
    repetition_penalty: float = 0.1
    penalty_decay: float = 0.99
    stop_on_boxed: bool = False
    stop_on_user: bool = False
    stop_check_every: int = 8
    stop_check_window: int = 96

    teacher_prompt_mode: str = "solution"
    distill_temperature: float = 1.0
    loss_type: str = "jsd"
    top_k_loss: int = 0
    jsd_token_clip: float = 0.5
    clip_mode: str = "clamp"
    min_clip_keep_frac: float = 0.05

    optimizer: str = "muon"
    lr: float = 5e-6
    adamw_fallback_lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    optimizer_eps: float = 1e-18
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-12
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_variant: str = "torch"

    log_interval: int = 1
    save_interval: int = 20
    save_rollouts: bool = True
    save_top_token_debug: int = 8


def set_seed(seed: int):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_decode_factory(tok):
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
    return safe_decode


def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_teacher_prompt(problem: str, solution: str, answer: str, mode: str = "solution") -> str:
    p = str(problem or "").strip()
    sol = str(solution or "").strip()
    ans = str(answer or "").strip()
    if not ans:
        ans = extract_gsm8k_final_answer(sol)
    mode = str(mode or "solution").lower()
    if mode == "answer":
        privileged = f"The verified final answer is {ans}."
    else:
        privileged = (
            "A verified reference solution is available:\n"
            f"{sol}\n"
            f"The verified final answer is {ans}."
        )
    return (
        f"User: {p}\n"
        f"{privileged}\n"
        f"Use the reference only as hidden guidance. Produce the normal student-style reasoning. "
        f"Put the final answer in \\boxed{{...}}, and make the final line contain only \\boxed{{...}}. think\n"
        f"Assistant: <think>\n"
    )


def pad_batch(seqs: List[List[int]], device: str, pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
    lens = [len(s) for s in seqs]
    T = max(1, max(lens) if lens else 1)
    x = torch.full((len(seqs), T), int(pad_id), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if s:
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return x, lens


def pack_by_token_budget(items: List[Dict[str, Any]], token_budget: int) -> List[List[Dict[str, Any]]]:
    if token_budget <= 0:
        return [[x] for x in items]
    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    used = 0
    for item in items:
        need = max(1, int(item.get("student_len", len(item["student_full_tokens"]))) - 1)
        if cur and used + need > int(token_budget):
            batches.append(cur)
            cur = []
            used = 0
        cur.append(item)
        used += need
    if cur:
        batches.append(cur)
    return batches


def distill_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    loss_type: str,
    top_k_loss: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-token loss and raw JSD diagnostic for [B,T,V] logits."""
    temp = max(float(temperature), 1e-6)
    s = student_logits.float() / temp
    t = teacher_logits.float() / temp
    loss_type = str(loss_type or "jsd").lower()
    k = int(top_k_loss)

    if k > 0 and k < int(s.size(-1)):
        union_idx = torch.cat(
            [
                torch.topk(s, k=k, dim=-1).indices,
                torch.topk(t, k=k, dim=-1).indices,
            ],
            dim=-1,
        )
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask.scatter_(-1, union_idx, True)
        s = s.masked_fill(~mask, -1e30)
        t = t.masked_fill(~mask, -1e30)

    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)
    s_prob = s_logp.exp()
    t_prob = t_logp.exp()
    m_prob = 0.5 * (s_prob + t_prob)
    m_logp = torch.log(m_prob.clamp_min(1e-30))
    jsd = 0.5 * (
        (t_prob * (t_logp - m_logp)).sum(dim=-1)
        + (s_prob * (s_logp - m_logp)).sum(dim=-1)
    )

    if loss_type == "teacher_kl":
        loss = (t_prob.detach() * (t_logp.detach() - s_logp)).sum(dim=-1)
    elif loss_type == "student_kl":
        loss = (s_prob * (s_logp - t_logp.detach())).sum(dim=-1)
    elif loss_type == "sym_kl":
        loss = 0.5 * (
            (t_prob.detach() * (t_logp.detach() - s_logp)).sum(dim=-1)
            + (s_prob * (s_logp - t_logp.detach())).sum(dim=-1)
        )
    elif loss_type == "jsd_teacher_grad":
        loss = 0.5 * (
            (t_prob.detach() * (t_logp.detach() - m_logp)).sum(dim=-1)
            + (s_prob * (s_logp - m_logp)).sum(dim=-1)
        )
    else:
        loss = jsd
    return loss.float(), jsd.float()


def apply_token_clip(
    loss: torch.Tensor,
    jsd: torch.Tensor,
    token_mask: torch.Tensor,
    cfg: OPSDConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clip = float(cfg.jsd_token_clip)
    mode = str(cfg.clip_mode or "clamp").lower()
    active = token_mask.bool()
    if clip <= 0 or mode == "none":
        return loss, active, jsd
    if mode == "mask":
        keep = active & (jsd <= clip)
        active_count = int(active.sum().item())
        keep_count = int(keep.sum().item())
        if active_count > 0 and keep_count < max(1, int(math.ceil(active_count * float(cfg.min_clip_keep_frac)))):
            vals = jsd[active]
            kth = max(1, int(math.ceil(active_count * float(cfg.min_clip_keep_frac))))
            thresh = torch.topk(vals, k=kth, largest=False).values[-1]
            keep = active & (jsd <= thresh)
        return loss, keep, jsd
    clipped_loss = torch.minimum(loss, torch.full_like(loss, clip))
    return clipped_loss, active, jsd


def build_optimizer(model: torch.nn.Module, cfg: OPSDConfig):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if cfg.optimizer == "muon":
        opt = Muon2DWithAdamWFallback(
            params,
            lr=float(cfg.lr),
            adam_lr=float(cfg.adamw_fallback_lr),
            betas=(float(cfg.beta1), float(cfg.beta2)),
            adam_eps=float(cfg.optimizer_eps),
            muon_momentum=float(cfg.muon_momentum),
            muon_nesterov=bool(cfg.muon_nesterov),
            muon_ns_steps=int(cfg.muon_ns_steps),
            muon_eps=float(cfg.muon_eps),
            muon_adjust_lr_fn=str(cfg.muon_adjust_lr_fn),
            muon_variant=str(cfg.muon_variant),
            muon_weight_decay=float(cfg.weight_decay),
            adam_weight_decay=float(cfg.weight_decay),
        )
        print(
            "Optimizer: Muon2DWithAdamWFallback "
            f"muon_tensors={len(opt.muon_params)} muon_params={sum(p.numel() for p in opt.muon_params)} "
            f"adam_tensors={len(opt.adam_params)} adam_params={sum(p.numel() for p in opt.adam_params)} "
            f"lr={cfg.lr} adam_lr={cfg.adamw_fallback_lr} betas=({cfg.beta1},{cfg.beta2}) "
            f"adam_eps={cfg.optimizer_eps} muon_eps={cfg.muon_eps} wd={cfg.weight_decay}",
            flush=True,
        )
        return opt
    opt = MemoryEfficientAdamW(
        [p for _, p in params],
        lr=float(cfg.lr),
        betas=(float(cfg.beta1), float(cfg.beta2)),
        eps=float(cfg.optimizer_eps),
        weight_decay=float(cfg.weight_decay),
        enabled=True,
    )
    print(
        f"Optimizer: MemoryEfficientAdamW params={sum(p.numel() for _, p in params)} "
        f"lr={cfg.lr} betas=({cfg.beta1},{cfg.beta2}) eps={cfg.optimizer_eps} wd={cfg.weight_decay}",
        flush=True,
    )
    return opt


class OPSDTrainerLocal:
    def __init__(self, cfg: OPSDConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.log_path = os.path.join(cfg.out_dir, "train.log")
        self.rollout_path = os.path.join(cfg.out_dir, "rollouts.jsonl")
        self.token_debug_path = os.path.join(cfg.out_dir, "token_debug.jsonl")
        self.tok = TRIE_TOKENIZER(cfg.tokenizer)
        self.encode = lambda s: self.tok.encode(s)
        self.decode = safe_decode_factory(self.tok)

        base_name, pth_path = normalize_model_arg(cfg.model)
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"Cannot find model pth: {pth_path}")
        teacher_arg = cfg.teacher_model or cfg.model
        _, teacher_pth = normalize_model_arg(teacher_arg)
        if not os.path.isfile(teacher_pth):
            raise FileNotFoundError(f"Cannot find teacher model pth: {teacher_pth}")

        self.log(f"CONFIG {json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True)}")
        self.log(f"loading student/train model: {pth_path}")
        self.model, self.model_args = load_train_model_rwkv7_cuda(
            pth_path, device=cfg.device, ctx_len=cfg.ctx_len, grad_cp=cfg.grad_cp
        )
        self.model.train()

        self.teacher_model = None
        if os.path.abspath(teacher_pth) != os.path.abspath(pth_path):
            self.log(f"loading fixed teacher model: {teacher_pth}")
            self.teacher_model, _ = load_train_model_rwkv7_cuda(
                teacher_pth, device=cfg.device, ctx_len=cfg.ctx_len, grad_cp=0
            )
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        self.log(f"loading rollout fp16 model: {base_name}")
        infer_model, _ = load_infer_model_fp16(base_name, device=cfg.device)
        infer_cfg = GRPOConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            use_rapid_sampling=cfg.use_rapid_sampling,
            presence_penalty=cfg.presence_penalty,
            repetition_penalty=cfg.repetition_penalty,
            penalty_decay=cfg.penalty_decay,
            mask_token0=False,
            stop_on_boxed=cfg.stop_on_boxed,
            stop_on_user=cfg.stop_on_user,
            stop_check_every=cfg.stop_check_every,
            stop_check_window=cfg.stop_check_window,
        )
        self.infer = FP16BatchInference(
            infer_model=infer_model,
            train_model=self.model,
            encode_fn=self.encode,
            decode_fn=self.decode,
            device=cfg.device,
            cfg=infer_cfg,
        )
        self.infer.sync_infer_weights(step=0, force=True)

        self.opt = build_optimizer(self.model, cfg)
        self.data = load_data(cfg.train_jsonl)
        if not self.data:
            raise RuntimeError(f"No training records loaded from {cfg.train_jsonl}")
        self.rng = random.Random(int(cfg.seed))
        self.log(f"loaded train records: {len(self.data)}")

    def log(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def sample_examples(self) -> List[Dict[str, Any]]:
        n = min(int(self.cfg.batch_prompts), len(self.data))
        return [dict(x) for x in self.rng.sample(self.data, n)]

    @torch.no_grad()
    def rollout(self, examples: List[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
        max_prompt_len = int(self.cfg.ctx_len) - int(self.cfg.max_new_tokens) - 4
        max_prompt_len = max(64, max_prompt_len)
        student_prompts = [build_prompt(ex.get("problem", "")) for ex in examples]
        student_prompt_tokens: List[List[int]] = []
        for p in student_prompts:
            ids = self.encode(p)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            student_prompt_tokens.append(ids)

        comp_tokens, old_logps, comp_texts, truncs = self.infer.generate_group_parallel(
            prompt_tokens_list=student_prompt_tokens,
            group_size=int(self.cfg.group_size),
            max_new_tokens=int(self.cfg.max_new_tokens),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            top_k=int(self.cfg.top_k),
            stop_on_think_close=False,
            stop_on_user=bool(self.cfg.stop_on_user),
            stop_on_boxed=bool(self.cfg.stop_on_boxed),
            stop_check_every=int(self.cfg.stop_check_every),
            stop_check_window=int(self.cfg.stop_check_window),
            presence_penalty=float(self.cfg.presence_penalty),
            frequency_penalty=float(self.cfg.repetition_penalty),
            penalty_decay=float(self.cfg.penalty_decay),
            rng_seed=int(self.cfg.seed) + int(step) * 100003,
        )

        trajs: List[Dict[str, Any]] = []
        for i, ex in enumerate(examples):
            sol = str(ex.get("solution", ""))
            ans = str(ex.get("answer", "")) or extract_gsm8k_final_answer(sol)
            teacher_prompt = build_teacher_prompt(
                ex.get("problem", ""),
                solution=sol,
                answer=ans,
                mode=self.cfg.teacher_prompt_mode,
            )
            t_ids = self.encode(teacher_prompt)
            if len(t_ids) > max_prompt_len:
                t_ids = t_ids[-max_prompt_len:]
            for g in range(int(self.cfg.group_size)):
                j = i * int(self.cfg.group_size) + g
                comp = [int(x) for x in comp_tokens[j]]
                if not comp:
                    continue
                sp = student_prompt_tokens[i]
                traj = {
                    "step": int(step),
                    "example_idx": int(i),
                    "group_idx": int(g),
                    "problem": str(ex.get("problem", "")),
                    "answer": ans,
                    "solution": sol,
                    "student_prompt": student_prompts[i],
                    "teacher_prompt": teacher_prompt,
                    "completion": comp_texts[j],
                    "truncated": bool(truncs[j]),
                    "old_logps": [float(x) for x in old_logps[j]],
                    "student_prompt_len": len(sp),
                    "teacher_prompt_len": len(t_ids),
                    "comp_len": len(comp),
                    "student_full_tokens": sp + comp,
                    "teacher_full_tokens": t_ids + comp,
                    "student_len": len(sp) + len(comp),
                    "teacher_len": len(t_ids) + len(comp),
                }
                trajs.append(traj)
                if self.cfg.save_rollouts:
                    out = dict(traj)
                    out.pop("student_full_tokens", None)
                    out.pop("teacher_full_tokens", None)
                    append_jsonl(self.rollout_path, out)
        return trajs

    def _forward_model(self, model: torch.nn.Module, seqs: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        padded, lens = pad_batch(seqs, device=self.cfg.device, pad_id=0)
        inp = padded[:, :-1].contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)
        if torch.is_tensor(logits) and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        return logits, lens

    def _debug_top_tokens(
        self,
        step: int,
        batch: List[Dict[str, Any]],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        jsd: torch.Tensor,
        token_mask: torch.Tensor,
    ):
        k = int(self.cfg.save_top_token_debug)
        if k <= 0:
            return
        cand = []
        active = token_mask.bool()
        for bi, tr in enumerate(batch):
            n = int(active[bi].sum().item())
            if n <= 0:
                continue
            vals = jsd[bi, :n].detach()
            take = min(k, int(vals.numel()))
            topv, topi = torch.topk(vals, k=take, largest=True)
            for v, ti in zip(topv.cpu().tolist(), topi.cpu().tolist()):
                cand.append((float(v), bi, int(ti), tr))
        cand.sort(key=lambda x: x[0], reverse=True)
        for v, bi, ti, tr in cand[:k]:
            s_lp = F.log_softmax(student_logits[bi, ti].float(), dim=-1)
            t_lp = F.log_softmax(teacher_logits[bi, ti].float(), dim=-1)
            s_top = torch.topk(s_lp, k=8)
            t_top = torch.topk(t_lp, k=8)
            tok_id = int(tr["student_full_tokens"][int(tr["student_prompt_len"]) + ti])
            append_jsonl(
                self.token_debug_path,
                {
                    "step": int(step),
                    "jsd": float(v),
                    "target_token_id": tok_id,
                    "target_text": self.decode([tok_id]),
                    "context_tail": self.decode(tr["student_full_tokens"][: int(tr["student_prompt_len"]) + ti][-96:]),
                    "student_top": [
                        {"id": int(idx), "text": self.decode([int(idx)]), "logp": float(lp)}
                        for lp, idx in zip(s_top.values.cpu().tolist(), s_top.indices.cpu().tolist())
                    ],
                    "teacher_top": [
                        {"id": int(idx), "text": self.decode([int(idx)]), "logp": float(lp)}
                        for lp, idx in zip(t_top.values.cpu().tolist(), t_top.indices.cpu().tolist())
                    ],
                },
            )

    def update_on_trajs(self, trajs: List[Dict[str, Any]], step: int) -> Dict[str, float]:
        if not trajs:
            return {"traj": 0.0, "tok": 0.0, "loss": 0.0}
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()
        self.opt.zero_grad(set_to_none=True)

        total_loss_num = 0.0
        total_loss_den = 0
        total_jsd = 0.0
        total_jsd_den = 0
        total_clipped = 0
        total_masked = 0
        total_tokens = 0
        token_budget = int(self.cfg.update_token_budget)
        if token_budget <= 0:
            token_budget = int(self.cfg.micro_batch_size) * int(self.cfg.max_new_tokens)
        batches = pack_by_token_budget(trajs, token_budget)
        total_update_tokens = sum(int(tr["comp_len"]) for tr in trajs)
        total_update_tokens = max(1, total_update_tokens)

        for batch in batches:
            student_seqs = [tr["student_full_tokens"] for tr in batch]
            teacher_seqs = [tr["teacher_full_tokens"] for tr in batch]
            s_logits_all, _ = self._forward_model(self.model, student_seqs)
            teacher_model = self.teacher_model if self.teacher_model is not None else self.model
            with torch.no_grad():
                t_logits_all, _ = self._forward_model(teacher_model, teacher_seqs)

            comp_lens = [int(tr["comp_len"]) for tr in batch]
            max_c = max(comp_lens)
            B = len(batch)
            V = int(s_logits_all.size(-1))
            s_seg = s_logits_all.new_zeros((B, max_c, V))
            t_seg = t_logits_all.new_zeros((B, max_c, V))
            tok_mask = torch.zeros((B, max_c), dtype=torch.bool, device=self.cfg.device)

            for bi, tr in enumerate(batch):
                c = int(tr["comp_len"])
                sp = int(tr["student_prompt_len"])
                tp = int(tr["teacher_prompt_len"])
                s_start = max(0, sp - 1)
                t_start = max(0, tp - 1)
                s_seg[bi, :c] = s_logits_all[bi, s_start : s_start + c]
                t_seg[bi, :c] = t_logits_all[bi, t_start : t_start + c]
                tok_mask[bi, :c] = True

            per_tok_loss, jsd = distill_logits(
                s_seg,
                t_seg,
                temperature=float(self.cfg.distill_temperature),
                loss_type=str(self.cfg.loss_type),
                top_k_loss=int(self.cfg.top_k_loss),
            )
            clipped_loss, keep_mask, raw_jsd = apply_token_clip(per_tok_loss, jsd, tok_mask, self.cfg)
            loss_den = int(keep_mask.sum().item())
            if loss_den <= 0:
                continue

            loss = clipped_loss[keep_mask].sum() / float(total_update_tokens)
            loss.backward()

            raw_active = tok_mask.bool()
            total_loss_num += float(clipped_loss[keep_mask].detach().sum().item())
            total_loss_den += loss_den
            total_jsd += float(raw_jsd[raw_active].detach().sum().item())
            total_jsd_den += int(raw_active.sum().item())
            total_clipped += int(((per_tok_loss > float(self.cfg.jsd_token_clip)) & raw_active).sum().item()) if self.cfg.jsd_token_clip > 0 else 0
            total_masked += int((raw_active & ~keep_mask).sum().item())
            total_tokens += int(raw_active.sum().item())
            self._debug_top_tokens(step, batch, s_seg.detach(), t_seg.detach(), raw_jsd.detach(), raw_active)

        grad_norm = 0.0
        if float(self.cfg.grad_clip) > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip)).item())
        self.opt.step()
        self.infer.sync_infer_weights(step=step, force=True)

        return {
            "traj": float(len(trajs)),
            "tok": float(total_tokens),
            "loss": float(total_loss_num / max(1, total_loss_den)),
            "raw_jsd": float(total_jsd / max(1, total_jsd_den)),
            "clip_frac": float(total_clipped / max(1, total_tokens)),
            "mask_frac": float(total_masked / max(1, total_tokens)),
            "grad": float(grad_norm),
        }

    def save(self, step: int):
        state = {n: p.detach().cpu() for n, p in self.model.state_dict().items()}
        ckpt_path = os.path.join(self.cfg.out_dir, f"ckpt_step{step}.pth")
        torch.save(
            {
                "time": now_str(),
                "step": int(step),
                "cfg": asdict(self.cfg),
                "model_state": state,
            },
            ckpt_path,
        )
        torch.save(state, os.path.join(self.cfg.out_dir, "latest_full_model.pth"))
        self.log(f"saved: {ckpt_path}")

    def train(self):
        for step in range(1, int(self.cfg.max_steps) + 1):
            t0 = time.time()
            examples = self.sample_examples()
            trajs = self.rollout(examples, step=step)
            stats = self.update_on_trajs(trajs, step=step)
            if step % int(self.cfg.log_interval) == 0:
                self.log(
                    f"[OPSD step {step}/{self.cfg.max_steps}] "
                    f"loss={stats['loss']:.6f} raw_jsd={stats['raw_jsd']:.6f} "
                    f"tok={int(stats['tok'])} traj={int(stats['traj'])} "
                    f"clip_frac={stats['clip_frac']:.4f} mask_frac={stats['mask_frac']:.4f} "
                    f"grad={stats['grad']:.6f} dt={time.time() - t0:.2f}s"
                )
            if int(self.cfg.save_interval) > 0 and step % int(self.cfg.save_interval) == 0:
                self.save(step)


def parse_args() -> OPSDConfig:
    ap = argparse.ArgumentParser(description="RWKV GSM8K OPSD local trainer")
    d = OPSDConfig()
    for field, val in asdict(d).items():
        arg = "--" + field
        if isinstance(val, bool):
            ap.add_argument(arg, action="store_true" if not val else "store_false", default=val)
        elif isinstance(val, int):
            ap.add_argument(arg, type=int, default=val)
        elif isinstance(val, float):
            ap.add_argument(arg, type=float, default=val)
        else:
            ap.add_argument(arg, type=str, default=val)
    args = ap.parse_args()
    cfg = OPSDConfig(**vars(args))
    if not cfg.out_dir:
        cfg.out_dir = os.path.join(REPO_DIR, f"out_opsd_gsm8k_{now_str()}")
    if cfg.optimizer not in ("adamw", "muon"):
        raise ValueError("--optimizer must be adamw or muon")
    if cfg.loss_type not in ("jsd", "jsd_teacher_grad", "teacher_kl", "student_kl", "sym_kl"):
        raise ValueError("--loss_type must be jsd/jsd_teacher_grad/teacher_kl/student_kl/sym_kl")
    if cfg.clip_mode not in ("clamp", "mask", "none"):
        raise ValueError("--clip_mode must be clamp/mask/none")
    if cfg.teacher_prompt_mode not in ("solution", "answer"):
        raise ValueError("--teacher_prompt_mode must be solution/answer")
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    trainer = OPSDTrainerLocal(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
