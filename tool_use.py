#!/usr/bin/env python3
"""
Pure baseline evaluator for RWKV tool-use data.

This script keeps the prompt format from the original `tool_use.py`, but removes
all GRPO / training logic. It evaluates a base RWKV model on the first N samples
from a JSONL file and reports answer accuracy using the same boxed-answer regex
judge used in the original script.
"""

import argparse
import ast
import atexit
import contextlib
import io
import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(REPO_DIR, "reference")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, REFERENCE_DIR)

HEAD_SIZE = 64
PROMPT_SUFFIX = "请将最终答案放在\\boxed{...}里，并且最终只给出\\boxed{...}这一行，不要输出多余内容。 think"
STOP_TOKENS = {0, 261, 24281}
TOOL_CALL_RE = re.compile(
    r"<tool_call(?:\s+name=\"([^\"]+)\")?\s*>(.*?)</tool_call>",
    re.S | re.I,
)
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_torch_cuda_arch_list(device: str) -> str:
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


def _ensure_cuda_toolkit_env() -> None:
    def _extract_nvcc_bin(raw: str) -> str:
        if not raw:
            return ""
        try:
            parts = shlex.split(raw)
        except Exception:
            parts = [raw]
        if not parts:
            return ""
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
        raise RuntimeError("Cannot find nvcc in PATH. CUDA toolkit is required.")

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
        raise RuntimeError(f"CUDA toolkit headers not found. Tried: {include_candidates}")

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["CUDACXX"] = nvcc_real

    def _prepend_unique_env_paths(name: str, paths: List[str]) -> None:
        cur = os.environ.get(name, "")
        cur_list = [x for x in cur.split(":") if x]
        out = []
        for p in paths + cur_list:
            if p and p not in out and len(p) <= 2048:
                out.append(p)
        os.environ[name] = ":".join(out)

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

    cxx = os.environ.get("CXX")
    if cxx:
        os.environ["CUDAHOSTCXX"] = cxx


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


def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def load_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int):
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
        grad_cp=0,
        train_type="fullstate",
        peft="none",
        my_testing="x070",
    )

    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(device)
    return model, args


def load_reference_model_fp16(base_name_no_pth: str, device: str = "cuda:0"):
    import types
    import importlib

    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(REPO_DIR, ".torch_extensions")

    rwkv7_fp16_mod = importlib.import_module("rwkv7_fp16")
    RWKV_x070 = rwkv7_fp16_mod.RWKV_x070

    infer_idx = _parse_cuda_index(device)
    if infer_idx is not None:
        torch.cuda.set_device(infer_idx)

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    model = RWKV_x070(args)
    return model, args


def build_prompt(problem: str) -> str:
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"{PROMPT_SUFFIX}\n"
        f"Assistant: <think>\n"
    )


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
    if str(inner).strip() in ("", "...", "…", "．．．"):
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
                content = text[start:i].strip()
                if content in ("...", "…", "．．．", ""):
                    return False
                return True
        i += 1
    return False


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
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def _normalize_final_answer_verl(final_answer: str) -> str:
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
    string = re.sub(r"[\.。,，;；:：!！?？]+$", "", string)
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
        s = str(s or "")
        return re.sub(r"(?<!\\frac\{)(\\?[A-Za-z]+)\s*/\s*([0-9]+)", r"\\frac{\1}{\2}", s)

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
    if tail is None:
        return None
    s = str(tail).lstrip()
    if not s:
        return None
    if s[0] in [":", "：", "="]:
        s = s[1:].lstrip()
        if not s:
            return None
    else:
        if s[0] in [".", "。", ",", "，", ";", "；", "!", "！", "?", "？"]:
            return None
        if not re.match(r"^[\-\+\(\[\{\\0-9a-zA-Z]", s):
            return None
    s = s.splitlines()[0]
    s = s.split("</think>")[0].strip()
    if not s:
        return None
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
                s = s[m.end():].lstrip(" :：=,-")
                lowered = s.lower()
                changed = True
                break
    if not s:
        return None
    if re.match(r"^(in\s+box|in\s+a\s+box|in\s+the\s+box|in\s+boxed(\s+form)?|with(in)?\s+\\?boxed|in\s+the\s+\\?boxed\{\}\s+notation|in\s+\\?boxed\{\}\s+notation)", s.lower()):
        return None
    boxed = extract_last_boxed(s)
    if boxed is not None and str(boxed).strip():
        return str(boxed).strip()
    s = re.split(r"\.\s+(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]
    s = re.split(r"[;,，；:：]\s*(?=[A-Za-z\u4e00-\u9fff])", s, maxsplit=1)[0]
    s = re.sub(r"[\.。,，;；:：!！?？]+$", "", s).strip()
    if not s:
        return None
    return s


def _extract_last_valid_answer_marker(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(re.finditer(r"(?i)\banswer\b", text))
    if not matches:
        return None
    candidates = []
    for m in matches:
        tail = text[m.end():]
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


def judge_prediction(pred_full_output: str, gt: str, truncated: bool = False) -> Dict[str, Any]:
    solution_str = str(pred_full_output or "")[-1200:]
    gt = str(gt or "")
    boxed = extract_last_boxed(solution_str)
    if boxed is not None and str(boxed).strip():
        extracted_answer = boxed
        extract_source = "boxed"
    else:
        marker_ans = _extract_last_valid_answer_marker(solution_str)
        if marker_ans is not None:
            extracted_answer = marker_ans
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
    if (not ok) and _is_numeric_only_text(gt_norm):
        gt_num = _extract_first_number(gt_norm)
        pred_num = _extract_first_number(extracted_answer)
        if gt_num is not None and pred_num is not None:
            ok = abs(gt_num - pred_num) <= 1e-9
    if truncated:
        ok = False
    return {
        "ok": bool(ok),
        "raw": extracted_answer,
        "extract_source": extract_source,
        "pred_norm": pred_norm,
        "gt_norm": gt_norm,
        "truncated_forced_zero": bool(truncated),
    }


def read_first_n_records(path: str, n: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def safe_decode(tok, ids: List[int]) -> str:
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


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_decay: float = 0.0
    stop_on_user: bool = True
    stop_on_boxed: bool = True
    stop_on_tool_call: bool = True
    stop_check_every: int = 1
    stop_check_window: int = 256
    micro_bsz: int = 960


def extract_last_tool_call(text: str) -> Optional[Dict[str, str]]:
    matches = list(TOOL_CALL_RE.finditer(str(text or "")))
    if not matches:
        return None
    m = matches[-1]
    tool_name = (m.group(1) or "").strip()
    inner = (m.group(2) or "").strip()
    code_match = CODE_FENCE_RE.search(inner)
    code = code_match.group(1).strip("\n") if code_match else ""
    if not tool_name:
        simple_name = inner.strip()
        if "\n" not in simple_name and "{" not in simple_name and len(simple_name) <= 128:
            tool_name = simple_name
    return {
        "tool_name": tool_name or "tool",
        "raw": m.group(0),
        "inner": inner,
        "code": code,
    }


def tool_call_complete(text: str) -> bool:
    return extract_last_tool_call(text) is not None


class StatefulPythonToolSession:
    _WORKER_CODE = r"""
import ast, contextlib, io, json, sys, traceback

NS = {"__name__": "__main__"}

def run_cell(code):
    out = io.StringIO()
    try:
        mod = ast.parse(code, mode="exec")
        body = list(mod.body)
        last_expr = None
        if body and isinstance(body[-1], ast.Expr):
            last_expr = ast.Expression(body[-1].value)
            body = body[:-1]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<tool>", "exec"), NS, NS)
            value = None
            if last_expr is not None:
                value = eval(compile(last_expr, "<tool>", "eval"), NS, NS)
        text = out.getvalue()
        if value is not None:
            if text and not text.endswith("\n"):
                text += "\n"
            text += repr(value)
        return {"ok": True, "output": text}
    except Exception:
        return {"ok": False, "output": traceback.format_exc()}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "close":
        print(json.dumps({"ok": True, "output": ""}, ensure_ascii=False), flush=True)
        break
    res = run_cell(str(req.get("code") or ""))
    print(json.dumps(res, ensure_ascii=False), flush=True)
"""

    def __init__(self, python_bin: str = sys.executable):
        self.python_bin = python_bin
        self.proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        if self._closed:
            return
        self.proc = subprocess.Popen(
            [self.python_bin, "-u", "-c", self._WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._queue.put(line.rstrip("\n"))
        self._queue.put(None)

    def execute(self, code: str, timeout: float) -> str:
        if self._closed:
            return "Tool session closed"
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        if self.proc is None or self.proc.stdin is None:
            return "Failed to start tool session"
        req = json.dumps({"code": str(code or "")}, ensure_ascii=False)
        try:
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
        except Exception:
            self._terminate()
            return "Failed to write to tool session"
        try:
            line = self._queue.get(timeout=max(0.1, float(timeout)))
        except queue.Empty:
            self._terminate()
            return "Timed out"
        if line is None:
            self._terminate()
            return "Tool session terminated unexpectedly"
        try:
            res = json.loads(line)
        except Exception:
            return str(line)
        return str(res.get("output") or "")

    def _terminate(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass

    def close(self) -> None:
        self._closed = True
        proc = self.proc
        if proc is not None and proc.stdin is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
        self._terminate()


class LocalToolExecutor:
    def __init__(self, timeout_sec: float = 20.0, parallelism: int = 8):
        self.timeout_sec = float(timeout_sec)
        self.parallelism = max(1, int(parallelism))
        self._sessions: Dict[int, StatefulPythonToolSession] = {}
        atexit.register(self.close)

    def _get_session(self, sample_idx: int) -> StatefulPythonToolSession:
        sess = self._sessions.get(int(sample_idx))
        if sess is None:
            sess = StatefulPythonToolSession()
            self._sessions[int(sample_idx)] = sess
        return sess

    def execute_one(self, sample_idx: int, tool_name: str, code: str) -> str:
        if str(tool_name or "").strip() != "stateful_python_code_exec":
            return f"Unsupported tool: {tool_name}"
        return self._get_session(sample_idx).execute(code=code, timeout=self.timeout_sec)

    def execute_many(self, calls: List[Tuple[int, str, str]]) -> Dict[int, str]:
        if not calls:
            return {}
        if len(calls) == 1:
            sample_idx, tool_name, code = calls[0]
            return {int(sample_idx): self.execute_one(sample_idx, tool_name, code)}
        out: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(self.parallelism, len(calls))) as ex:
            futs = {
                ex.submit(self.execute_one, sample_idx, tool_name, code): int(sample_idx)
                for sample_idx, tool_name, code in calls
            }
            for fut, sample_idx in futs.items():
                try:
                    out[sample_idx] = fut.result()
                except Exception as exc:
                    out[sample_idx] = f"Tool execution failed: {exc}"
        return out

    def close(self) -> None:
        for sess in self._sessions.values():
            try:
                sess.close()
            except Exception:
                pass
        self._sessions.clear()


class RWKVEvaluator:
    def __init__(self, model: torch.nn.Module, tok, device: str, gen_cfg: GenerationConfig):
        self.model = model
        self.tok = tok
        self.device = device
        self.device_index = _parse_cuda_index(device)
        self.gen_cfg = gen_cfg
        self.ctx_len = int(getattr(getattr(model, "args", None), "ctx_len", 4096))

    def encode(self, text: str) -> List[int]:
        return [int(x) for x in self.tok.encode(text)]

    def decode(self, ids: List[int]) -> str:
        return safe_decode(self.tok, ids)

    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
        lens = [len(s) for s in seqs]
        T = max(1, max(lens) if lens else 1)
        x = torch.full((len(seqs), T), int(pad_id), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return x, lens

    def _sample_from_logits(
        self,
        logits_2d: torch.Tensor,
        batch_indices: List[int],
        rep_counts: List[Dict[int, float]],
        sample_gen: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = logits_2d.float()
        apply_penalty = (self.gen_cfg.presence_penalty != 0.0) or (self.gen_cfg.frequency_penalty != 0.0)
        if apply_penalty:
            for li, gi in enumerate(batch_indices):
                cnt = rep_counts[gi]
                if not cnt:
                    continue
                for tok_id, cval in cnt.items():
                    if 0 <= int(tok_id) < x.size(-1):
                        x[li, int(tok_id)] -= float(self.gen_cfg.presence_penalty + self.gen_cfg.frequency_penalty * cval)

        if self.gen_cfg.temperature <= 0:
            tok = torch.argmax(x, dim=-1)
            logp_all = F.log_softmax(x, dim=-1)
            lp = logp_all.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            return tok.long(), lp.float()

        x = x / float(self.gen_cfg.temperature)
        V = x.size(-1)
        k_cap = 0
        if self.gen_cfg.top_k and int(self.gen_cfg.top_k) > 0:
            k_cap = int(min(int(self.gen_cfg.top_k), V))
        elif self.gen_cfg.top_p and 0.0 < float(self.gen_cfg.top_p) < 1.0:
            k_cap = int(min(2048, V))

        if k_cap > 0:
            topv, topi = torch.topk(x, k=k_cap, dim=-1)
            if self.gen_cfg.top_p and 0.0 < float(self.gen_cfg.top_p) < 1.0:
                probs = F.softmax(topv, dim=-1)
                cdf = torch.cumsum(probs, dim=-1)
                keep = cdf <= float(self.gen_cfg.top_p)
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

    @torch.no_grad()
    def generate_batch(self, prompt_tokens_list: List[List[int]], seed: int) -> Tuple[List[str], List[bool]]:
        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        self.model.eval()

        full_tokens = []
        for p in prompt_tokens_list:
            base = [int(x) for x in p][-self.ctx_len:]
            if not base:
                base = [0]
            full_tokens.append(list(base))

        B = len(full_tokens)
        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        active = [True for _ in range(B)]
        truncated = [False for _ in range(B)]
        apply_penalty = (self.gen_cfg.presence_penalty != 0.0) or (self.gen_cfg.frequency_penalty != 0.0)
        rep_counts: List[Dict[int, float]] = [{} for _ in range(B)] if apply_penalty else [{} for _ in range(B)]
        sample_gen = None
        if self.gen_cfg.temperature > 0:
            sample_gen = torch.Generator(device=self.device)
            sample_gen.manual_seed(int(seed))

        for t in range(int(self.gen_cfg.max_new_tokens)):
            active_idx = [i for i, flag in enumerate(active) if flag]
            if not active_idx:
                break

            logits_chunks = []
            micro_bsz = max(1, int(self.gen_cfg.micro_bsz))
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
                        take_idx = torch.tensor([max(0, int(l) - 1) for l in lens], dtype=torch.long, device=inp.device)
                        batch_idx = torch.arange(inp.size(0), dtype=torch.long, device=inp.device)
                        logits_sub = logits_full[batch_idx, take_idx, :]
                logits_chunks.append(logits_sub)
            logits_last = torch.cat(logits_chunks, dim=0) if len(logits_chunks) > 1 else logits_chunks[0]
            tok_local, _ = self._sample_from_logits(logits_last, active_idx, rep_counts, sample_gen)
            tok_cpu = tok_local.detach().cpu().tolist()

            for li, gi in enumerate(active_idx):
                if not active[gi]:
                    continue
                tok = int(tok_cpu[li])
                full_tokens[gi].append(tok)
                comp_tokens[gi].append(tok)
                if apply_penalty:
                    cnt = rep_counts[gi]
                    if self.gen_cfg.penalty_decay not in (0.0, 1.0):
                        for k in list(cnt.keys()):
                            nv = float(cnt[k]) * float(self.gen_cfg.penalty_decay)
                            if nv < 1e-5:
                                del cnt[k]
                            else:
                                cnt[k] = nv
                    cnt[tok] = float(cnt.get(tok, 0.0) + 1.0)
                if tok in STOP_TOKENS:
                    active[gi] = False

            if (self.gen_cfg.stop_on_user or self.gen_cfg.stop_on_boxed or self.gen_cfg.stop_on_tool_call) and (t % max(1, self.gen_cfg.stop_check_every) == 0):
                for gi in active_idx:
                    if not active[gi]:
                        continue
                    w = comp_tokens[gi][-self.gen_cfg.stop_check_window:] if self.gen_cfg.stop_check_window > 0 else comp_tokens[gi]
                    s = self.decode(w)
                    if self.gen_cfg.stop_on_tool_call and tool_call_complete(s):
                        active[gi] = False
                        continue
                    if self.gen_cfg.stop_on_boxed and boxed_complete(s):
                        active[gi] = False
                        continue
                    if self.gen_cfg.stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[gi] = False
                        continue

        for i in range(B):
            if active[i]:
                truncated[i] = True

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_text, truncated


class RWKVReferenceEvaluator:
    def __init__(self, model, tok, device: str, gen_cfg: GenerationConfig):
        self.model = model
        self.tok = tok
        self.device = device
        self.device_index = _parse_cuda_index(device)
        self.gen_cfg = gen_cfg
        self.ctx_len = 8192

    def encode(self, text: str) -> List[int]:
        return [int(x) for x in self.tok.encode(text)]

    def decode(self, ids: List[int]) -> str:
        return safe_decode(self.tok, ids)

    def _sample_from_logits(
        self,
        logits_2d: torch.Tensor,
        rep_counts: List[Dict[int, float]],
        sample_gen: Optional[torch.Generator],
    ) -> torch.Tensor:
        x = logits_2d.float()
        apply_penalty = (self.gen_cfg.presence_penalty != 0.0) or (self.gen_cfg.frequency_penalty != 0.0)
        if apply_penalty:
            for i, cnt in enumerate(rep_counts):
                if not cnt:
                    continue
                for tok_id, cval in cnt.items():
                    if 0 <= int(tok_id) < x.size(-1):
                        x[i, int(tok_id)] -= float(self.gen_cfg.presence_penalty + self.gen_cfg.frequency_penalty * cval)

        if self.gen_cfg.temperature <= 0:
            return torch.argmax(x, dim=-1).long()

        x = x / float(self.gen_cfg.temperature)
        V = x.size(-1)
        k_cap = 0
        if self.gen_cfg.top_k and int(self.gen_cfg.top_k) > 0:
            k_cap = int(min(int(self.gen_cfg.top_k), V))
        elif self.gen_cfg.top_p and 0.0 < float(self.gen_cfg.top_p) < 1.0:
            k_cap = int(min(2048, V))

        if k_cap > 0:
            topv, topi = torch.topk(x, k=k_cap, dim=-1)
            if self.gen_cfg.top_p and 0.0 < float(self.gen_cfg.top_p) < 1.0:
                probs = F.softmax(topv, dim=-1)
                cdf = torch.cumsum(probs, dim=-1)
                keep = cdf <= float(self.gen_cfg.top_p)
                keep[:, 0] = True
                topv = topv.masked_fill(~keep, -1e30)
            probs = F.softmax(topv, dim=-1)
            if sample_gen is None:
                pick = torch.multinomial(probs, 1).squeeze(-1)
            else:
                pick = torch.multinomial(probs, 1, generator=sample_gen).squeeze(-1)
            return topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1).long()

        probs = F.softmax(x, dim=-1)
        if sample_gen is None:
            return torch.multinomial(probs, 1).squeeze(-1).long()
        return torch.multinomial(probs, 1, generator=sample_gen).squeeze(-1).long()

    @torch.no_grad()
    def generate_batch(self, prompt_tokens_list: List[List[int]], seed: int) -> Tuple[List[str], List[bool]]:
        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        if self.gen_cfg.temperature > 0:
            torch.cuda.manual_seed(int(seed))

        full_tokens = []
        for p in prompt_tokens_list:
            base = [int(x) for x in p][-self.ctx_len:]
            if not base:
                base = [0]
            full_tokens.append(list(base))

        B = len(full_tokens)
        if B == 0:
            return [], []

        state = self.model.generate_zero_state(B)
        last_logits = self.model.forward_batch(full_tokens, state, full_output=False)
        if torch.is_tensor(last_logits) and last_logits.dim() == 3:
            last_logits = last_logits[:, -1, :]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        active = [True for _ in range(B)]
        truncated = [False for _ in range(B)]
        apply_penalty = (self.gen_cfg.presence_penalty != 0.0) or (self.gen_cfg.frequency_penalty != 0.0)
        rep_counts: List[Dict[int, float]] = [{} for _ in range(B)] if apply_penalty else [{} for _ in range(B)]
        sample_gen = None
        if self.gen_cfg.temperature > 0:
            sample_gen = torch.Generator(device=self.device)
            sample_gen.manual_seed(int(seed))

        for t in range(int(self.gen_cfg.max_new_tokens)):
            if not any(active):
                break

            token_ids = self._sample_from_logits(last_logits, rep_counts, sample_gen)
            token_ids = token_ids.long()
            active_mask = torch.tensor(active, device=token_ids.device, dtype=torch.bool)
            token_ids = torch.where(active_mask, token_ids, torch.zeros_like(token_ids))
            tok_cpu = token_ids.detach().cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                tok = int(tok_cpu[i])
                comp_tokens[i].append(tok)
                if apply_penalty:
                    cnt = rep_counts[i]
                    if self.gen_cfg.penalty_decay not in (0.0, 1.0):
                        for k in list(cnt.keys()):
                            nv = float(cnt[k]) * float(self.gen_cfg.penalty_decay)
                            if nv < 1e-5:
                                del cnt[k]
                            else:
                                cnt[k] = nv
                    cnt[tok] = float(cnt.get(tok, 0.0) + 1.0)

            for i in range(B):
                if active[i] and int(tok_cpu[i]) in STOP_TOKENS:
                    active[i] = False

            if (self.gen_cfg.stop_on_user or self.gen_cfg.stop_on_boxed or self.gen_cfg.stop_on_tool_call) and (t % max(1, self.gen_cfg.stop_check_every) == 0):
                for i in range(B):
                    if not active[i]:
                        continue
                    w = comp_tokens[i][-self.gen_cfg.stop_check_window:] if self.gen_cfg.stop_check_window > 0 else comp_tokens[i]
                    s = self.decode(w)
                    if self.gen_cfg.stop_on_tool_call and tool_call_complete(s):
                        active[i] = False
                        continue
                    if self.gen_cfg.stop_on_boxed and boxed_complete(s):
                        active[i] = False
                        continue
                    if self.gen_cfg.stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[i] = False
                        continue

            step_tokens_batch = [int(x) for x in tok_cpu]
            last_logits = self.model.forward_batch(step_tokens_batch, state, full_output=False)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        for i in range(B):
            if active[i]:
                truncated[i] = True

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_text, truncated


@dataclass
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    truncated: int
    boxed_predictions: int
    elapsed_sec: float
    model_path: str
    data_path: str
    num_samples: int
    temperature: float
    max_new_tokens: int
    tools_executed: int
    tool_rounds: int


def generate_with_tools(
    evaluator,
    prompts: List[str],
    seed: int,
    tool_executor: LocalToolExecutor,
    max_rounds: int,
) -> Tuple[List[str], List[bool], List[int]]:
    contexts = [str(p) for p in prompts]
    final_outputs = ["" for _ in prompts]
    truncated = [False for _ in prompts]
    tool_rounds = [0 for _ in prompts]
    active = [True for _ in prompts]

    for round_idx in range(max(1, int(max_rounds))):
        active_idx = [i for i, flag in enumerate(active) if flag]
        if not active_idx:
            break
        prompt_tokens = [evaluator.encode(contexts[i]) for i in active_idx]
        completions, truncated_flags = evaluator.generate_batch(prompt_tokens, seed=seed + round_idx * 100003)

        tool_calls: List[Tuple[int, str, str]] = []
        for local_i, sample_idx in enumerate(active_idx):
            completion = completions[local_i]
            contexts[sample_idx] += completion
            final_outputs[sample_idx] += completion
            truncated[sample_idx] = bool(truncated_flags[local_i])
            call = extract_last_tool_call(completion)
            if call is None or not call.get("code"):
                active[sample_idx] = False
                continue
            tool_calls.append((sample_idx, call.get("tool_name", "tool"), call.get("code", "")))

        if not tool_calls:
            continue

        tool_outputs = tool_executor.execute_many(tool_calls)
        for sample_idx, tool_name, _code in tool_calls:
            out = str(tool_outputs.get(int(sample_idx), ""))
            if not out.endswith("\n"):
                out += "\n"
            contexts[sample_idx] += f"Tool: {out}Assistant: "
            final_outputs[sample_idx] += f"Tool: {out}"
            tool_rounds[sample_idx] += 1
            truncated[sample_idx] = False

    return final_outputs, truncated, tool_rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/rwkv7-g1c-2.9b-20251231-ctx8192.pth")
    ap.add_argument("--data_path", type=str, default="tool_sft_test.jsonl")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--penalty_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=960)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--backend", type=str, choices=["reference", "train"], default="reference")
    ap.add_argument("--disable_tools", action="store_true")
    ap.add_argument("--tool_max_rounds", type=int, default=8)
    ap.add_argument("--tool_timeout", type=float, default=20.0)
    ap.add_argument("--tool_parallelism", type=int, default=16)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    os.environ["RWKV_MODEL_FP32_FORWARD"] = "0" if args.dtype == "bf16" else "1"

    _ensure_cuda_toolkit_env()
    safe_arch = _resolve_torch_cuda_arch_list(args.device)
    if safe_arch and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch

    out_dir = args.out_dir.strip() or f"out_eval_tool_use_baseline_{now_str()}"
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "predictions.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    from utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)

    _, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    print(f"[data] reading samples [{args.start_index}, {args.start_index + args.num_samples}) from {args.data_path}", flush=True)
    records = read_first_n_records(args.data_path, int(args.start_index) + int(args.num_samples))[int(args.start_index):]
    if not records:
        raise RuntimeError("No records loaded.")
    print(f"[data] loaded {len(records)} records", flush=True)

    print(f"[model] loading {pth_path} on {args.device} (backend={args.backend})", flush=True)
    base_name, _ = normalize_model_arg(args.model)
    if args.backend == "reference":
        model, _ = load_reference_model_fp16(base_name, device=args.device)
        evaluator_cls = RWKVReferenceEvaluator
    else:
        model, _ = load_model_rwkv7_cuda(pth_path, device=args.device, ctx_len=int(args.ctx_len))
        if args.dtype == "bf16":
            model = model.to(dtype=torch.bfloat16)
        model.eval()
        evaluator_cls = RWKVEvaluator

    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        presence_penalty=float(args.presence_penalty),
        frequency_penalty=float(args.frequency_penalty),
        penalty_decay=float(args.penalty_decay),
        stop_on_tool_call=not bool(args.disable_tools),
        micro_bsz=max(1, int(args.batch_size)),
    )
    evaluator = evaluator_cls(model=model, tok=tok, device=args.device, gen_cfg=gen_cfg)
    tool_executor = None if args.disable_tools else LocalToolExecutor(
        timeout_sec=float(args.tool_timeout),
        parallelism=int(args.tool_parallelism),
    )

    total = 0
    correct = 0
    truncated_n = 0
    boxed_n = 0
    tools_executed = 0
    tool_rounds_total = 0
    t0 = time.time()

    with open(results_path, "w", encoding="utf-8") as wf:
        for chunk_start in range(0, len(records), int(args.batch_size)):
            chunk = records[chunk_start: chunk_start + int(args.batch_size)]
            prompts = [build_prompt(str(rec.get("problem", ""))) for rec in chunk]
            if tool_executor is None:
                prompt_tokens = [evaluator.encode(p) for p in prompts]
                completions, truncated_flags = evaluator.generate_batch(prompt_tokens, seed=args.seed + chunk_start)
                tool_rounds_chunk = [0 for _ in chunk]
            else:
                completions, truncated_flags, tool_rounds_chunk = generate_with_tools(
                    evaluator=evaluator,
                    prompts=prompts,
                    seed=args.seed + chunk_start,
                    tool_executor=tool_executor,
                    max_rounds=int(args.tool_max_rounds),
                )

            for local_idx, (rec, prompt, completion, truncated, tool_rounds_one) in enumerate(
                zip(chunk, prompts, completions, truncated_flags, tool_rounds_chunk)
            ):
                judge = judge_prediction(completion, str(rec.get("expected_answer", "")), truncated=bool(truncated))
                total += 1
                correct += int(judge["ok"])
                truncated_n += int(bool(truncated))
                boxed_n += int(extract_last_boxed(completion) is not None)
                tools_executed += int(tool_rounds_one)
                tool_rounds_total += int(tool_rounds_one)

                row = {
                    "index": chunk_start + local_idx,
                    "uuid": rec.get("uuid"),
                    "problem": rec.get("problem"),
                    "expected_answer": rec.get("expected_answer"),
                    "prompt": prompt,
                    "completion": completion,
                    "predicted_answer": judge["raw"],
                    "extract_source": judge["extract_source"],
                    "ok": judge["ok"],
                    "truncated": bool(truncated),
                    "tool_usage": rec.get("tool_usage"),
                    "tool_rounds": int(tool_rounds_one),
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            acc = correct / max(1, total)
            print(
                f"[eval] {total}/{len(records)} done | acc={acc:.4f} | trunc={truncated_n} | boxed={boxed_n} | tools={tools_executed} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    summary = EvalSummary(
        total=total,
        correct=correct,
        accuracy=(correct / max(1, total)),
        truncated=truncated_n,
        boxed_predictions=boxed_n,
        elapsed_sec=elapsed,
        model_path=pth_path,
        data_path=args.data_path,
        num_samples=len(records),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        tools_executed=int(tools_executed),
        tool_rounds=int(tool_rounds_total),
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), flush=True)
    print(f"[out] predictions: {results_path}", flush=True)
    print(f"[out] summary: {summary_path}", flush=True)
    if tool_executor is not None:
        tool_executor.close()


if __name__ == "__main__":
    main()
