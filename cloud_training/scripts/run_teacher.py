"""run_teacher.py —— 启动 Teacher 模型 OpenAI 兼容服务(llama_cpp.server)。

用途:把 gen_products / gen_test_samples 等脚本与模型推理解耦,支持 HTTP 远程
调用 + 断线重连续传;便于监控与运维。

============================================================
★ 重要硬件约束(必读) ★
本机为 H200 MIG 2g.35gb(约 35GB 显存),llama_cpp 0.3.35:
  1. Llama 无 n_parallel / n_slots 参数 -> 单实例单 slot 串行推理,
     server 端并发请求会**排队串行**,吞吐 ≈ 单进程推理,不线性加速。
  2. 多实例会各自 cudaMalloc 17GB 权重 -> 2 实例即 OOM(>35GB)。
因此本脚本默认只起 **1 个实例**。client 端"并发"仅用于解耦/容错,
真正提速靠:消除 JSON 解析重试 + 合理 batch 大小(见 llm_client / gen_products)。
============================================================

OOM 保护:根据 n_ctx 估算 KV cache 显存,超过阈值自动降级 n_ctx。
模型权重 ~17GB(Q4_K_XL),MIG 35GB,留余量给 KV/激活/CUDA context。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings

_ROOT = Path(__file__).resolve().parents[2]  # 仓库根
DEFAULT_MODEL = str(_ROOT / "models" / "Qwen3.8-27B-UD-Q4_K_XL.gguf")

# 显存预算(字节)。MIG 2g.35gb ≈ 35GB。权重 ~17GB,fixed 开销(CUDA ctx/激活) ~4GB,
# 留给 KV cache 的预算 = TOTAL - WEIGHT - FIXED。
VRAM_TOTAL = 35 * 1024**3
VRAM_WEIGHT = 18 * 1024**3   # Q4_K_XL 实际略 >17GB,留余量
VRAM_FIXED = 4 * 1024**3
VRAM_KV_BUDGET = VRAM_TOTAL - VRAM_WEIGHT - VRAM_FIXED  # ≈ 13GB


def estimate_kv_bytes(n_ctx: int, n_layers: int = 40, n_kv_heads: int = 8,
                      head_dim: int = 128, dtype_bytes: int = 2) -> int:
    """粗略估算单序列 KV cache 显存。

    KV = 2(K,V) * n_layers * n_ctx * n_kv_heads * head_dim * dtype_bytes
    Qwen3.8-27B 近似:40 层 / 8 KV heads / 128 head_dim(Q4 下 KV 多为 fp16)。
    """
    return 2 * n_layers * n_ctx * n_kv_heads * head_dim * dtype_bytes


def pick_safe_n_ctx(requested: int) -> int:
    """在 KV 预算内选择不超过 requested 的最大 n_ctx(2 的幂对齐回退)。"""
    candidates = [8192, 4096, 2048, 1024]
    for c in candidates:
        if c > requested:
            continue
        if estimate_kv_bytes(c) <= VRAM_KV_BUDGET:
            return c
    # 全部超预算则取最小
    return 1024


def main():
    ap = argparse.ArgumentParser(description="Teacher LLM OpenAI-compatible server")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-ctx", type=int, default=4096,
                    help="目标上下文长度;超过 KV 预算会自动降级")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--flash-attn", action="store_true",
                    help="若构建支持则启用(flash attention 省显存/提速)")
    args = ap.parse_args()

    safe_ctx = pick_safe_n_ctx(args.n_ctx)
    if safe_ctx != args.n_ctx:
        print(f"[run_teacher] n_ctx {args.n_ctx} 超 KV 预算, 自动降级为 {safe_ctx} "
              f"(KV≈{estimate_kv_bytes(safe_ctx)/1024**3:.1f}GB, 预算≈"
              f"{VRAM_KV_BUDGET/1024**3:.1f}GB)", flush=True)
    else:
        print(f"[run_teacher] n_ctx={safe_ctx} (KV≈"
              f"{estimate_kv_bytes(safe_ctx)/1024**3:.1f}GB)", flush=True)

    settings = Settings(
        model=args.model,
        host=args.host,
        port=args.port,
        n_ctx=safe_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ubatch=min(args.n_batch, 512),
        offload_kqv=True,
        verbose=False,
    )

    print(f"[run_teacher] loading {args.model} on CUDA(n_gpu_layers="
          f"{args.n_gpu_layers}) ...", flush=True)
    app = create_app(settings)

    # 关键:server 端关闭 Qwen3 thinking(等价进程内 _patch_no_think)。
    # 该 GGUF 的 chat template 不认 chat_template_kwargs,必须直接 patch
    # 底层 Llama 实例的 chat_template 文本(恒关 thinking 分支)。
    from llama_cpp.server import app as _srv_app
    _llama = _srv_app._llama_proxy._current_model
    if getattr(_llama, "chat_template", None):
        t = _llama.chat_template
        t = t.replace("{%- if enable_thinking is defined and enable_thinking is false %}",
                      "{%- if true %}")
        t = t.replace("{%- if enable_thinking is defined and enable_thinking %}",
                      "{%- if false %}")
        _llama.chat_template = t
        print("[run_teacher] patched chat_template -> no-think (server side)",
              flush=True)
    else:
        print("[run_teacher] WARN: chat_template 为空,no-think patch 跳过",
              flush=True)

    import uvicorn
    # 单 worker(单 slot 串行);workers>1 会起多进程各自占 GPU -> OOM,故固定 1
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                log_level="info")


if __name__ == "__main__":
    main()
