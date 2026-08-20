"""llm_client.py —— Teacher 模型(Qwen3.8-27B GGUF)统一客户端。

推理引擎:llama-cpp-python 0.3.35(CUDA,MIG)。替代 task.md 原定的 Ollama
(决策见 decisions.log 2026-08-19)。

核心能力(D0 冒烟已验证):
1. thinking 显式关闭(硬约束 3):加载后替换 GGUF 内嵌 chat template 的
   enable_thinking 分支,generation prompt 恒注入空思考段 <think>\\n\\n</think>。
2. tools 参数透传给模板渲染(Qwen 原生工具协议生效),模型输出
   <tool_call> XML,由 parse_qwen_tool_calls 转为 OpenAI tool_calls 格式。
3. extract_json_array 容错提取(PRD 17 5.3.2 同款 + <think> 剥离双保险)。

随机种子由调用方脚本负责固定(本模块不引入随机性)。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

# 动态定位仓库根目录:本文件位于 <root>/cloud_training/scripts/llm_client.py
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_ROOT / "models" / "Qwen3.8-27B-UD-Q4_K_XL.gguf")
MAX_RETRIES = 3

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """剥离思考段(双保险:模板已关 thinking,正常情况无此段)"""
    return THINK_RE.sub("", text).strip()


def parse_qwen_tool_calls(text: str) -> list[dict]:
    """把 Qwen 原生 <tool_call> XML 输出解析为 OpenAI tool_calls 列表。

    返回 [{"id": "call_001", "type": "function",
           "function": {"name": ..., "arguments": {...}}}, ...]
    """
    calls = []
    for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
        name, body = m.group(1).strip(), m.group(2)
        args = {}
        for pm in PARAM_RE.finditer(body):
            val = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[pm.group(1).strip()] = val
        calls.append({
            "id": f"call_{i + 1:03d}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return calls


def extract_json_array(text: str) -> list:
    """容错提取 JSON 数组:剥离 think/markdown 代码块后取最后一个 [...]"""
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    matches = re.findall(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if not matches:
        return []
    try:
        return json.loads(matches[-1])
    except (json.JSONDecodeError, ValueError):
        return []


def extract_json_object(text: str) -> dict | None:
    """容错提取单个 JSON 对象(teacher 出题等场景)"""
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for m in reversed(matches):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class TeacherLLM:
    """进程内单例 Teacher。单序列同步推理(~13 tok/s);
    高并发场景请用 llama-server OpenAI 兼容服务(见 run_teacher.py)。"""

    def __init__(self, model_path: str = DEFAULT_MODEL, n_ctx: int = 8192,
                 n_gpu_layers: int = -1, verbose: bool = False):
        self.llm = Llama(
            model_path=model_path, n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx, n_batch=512, verbose=verbose,
        )
        self._patch_no_think()
        self.n_calls = 0

    def _patch_no_think(self) -> None:
        """覆盖 _chat_handlers['chat_template.default'] 为 no-think 模板。

        照抄 llama.py L502-536 构造方式(含 stop_token_ids,缺失会导致
        空 stop 序列瞬间截断输出——D0 实测踩坑)。
        """
        tpl = self.llm.metadata["tokenizer.chat_template"]
        tpl2 = tpl.replace(
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{%- if true %}",
        ).replace(
            "{%- if enable_thinking is undefined or enable_thinking is true %}",
            "{%- if false %}",
        )
        assert tpl2 != tpl, "chat template patch failed: pattern not found"
        eos_token_id = self.llm.token_eos()
        bos_token_id = self.llm.token_bos()
        eos_token = (self.llm._model.token_get_text(eos_token_id)
                     if eos_token_id != -1 else "")
        bos_token = (self.llm._model.token_get_text(bos_token_id)
                     if bos_token_id != -1 else "")
        self.llm._chat_handlers["chat_template.default"] = Jinja2ChatFormatter(
            template=tpl2, eos_token=eos_token, bos_token=bos_token,
            stop_token_ids=[eos_token_id],
        ).to_chat_handler()

    def chat(self, messages: list, tools: list | None = None,
             temperature: float = 0.7, max_tokens: int = 2048) -> dict:
        """一次对话调用。返回 {"content": str, "tool_calls": [...]。

        tool_calls 为 OpenAI 格式(经 parse_qwen_tool_calls);无调用时为 []。
        content 已剥离 <think> 段。
        """
        self.n_calls += 1
        kwargs = dict(
            messages=messages, temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        out = self.llm.create_chat_completion(**kwargs)
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        content = strip_think(content)
        calls = parse_qwen_tool_calls(content)
        return {"content": content, "tool_calls": calls}

    def chat_json_array(self, prompt: str, temperature: float = 0.9,
                        max_tokens: int = 4096, retries: int = MAX_RETRIES) -> list:
        """生成 JSON 数组(gen_products 等),失败重试(指数退避),终败返回 []"""
        for attempt in range(retries):
            out = self.chat([{"role": "user", "content": prompt}],
                            temperature=temperature, max_tokens=max_tokens)
            arr = extract_json_array(out["content"])
            if arr:
                return arr
            print(f"[llm_client] attempt {attempt + 1}/{retries} "
                  f"json parse failed, retrying...", flush=True)
            time.sleep(2 ** attempt)
        return []


# cloud_training/config/tools/tools.json(单一源)
_DEFAULT_TOOLS = str(_ROOT / "cloud_training" / "config" / "tools" / "tools.json")


def load_tools_schema(path: str = _DEFAULT_TOOLS) -> list:
    """读取 11 工具契约(单一源)"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
