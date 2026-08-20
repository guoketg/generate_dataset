"""start_teacher_vllm.py — 使用 vLLM 启动 Teacher 模型 OpenAI 兼容服务

用途：启动 qwen3.8:27b Q4 量化模型，提供 OpenAI 兼容 API 用于轨迹采样。

硬件约束：
- H200 MIG 2g.35gb（约 35GB 显存）
- 模型权重 ~17GB（Q4_K_XL）
- KV cache + 激活 + CUDA context 需要额外空间

使用方式：
    python start_teacher_vllm.py --model ../models/Qwen3.8-27B-UD-Q4_K_XL.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # 仓库根
DEFAULT_MODEL = str(_ROOT / "models" / "Qwen3.8-27B-UD-Q4_K_XL.gguf")


def main():
    ap = argparse.ArgumentParser(description="Teacher LLM vLLM OpenAI-compatible server")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF 模型路径")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--port", type=int, default=8000, help="监听端口")
    ap.add_argument("--max-model-len", type=int, default=8192, help="最大上下文长度")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU 显存利用率")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size", help="张量并行数")
    ap.add_argument("--dtype", default="auto", help="数据类型")
    ap.add_argument("--quantization", default="gguf", help="量化方式")
    args = ap.parse_args()

    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[start_teacher_vllm] 启动 vLLM 服务", flush=True)
    print(f"  模型: {args.model}", flush=True)
    print(f"  地址: {args.host}:{args.port}", flush=True)
    print(f"  最大上下文: {args.max_model_len}", flush=True)
    print(f"  GPU 显存利用率: {args.gpu_memory_utilization}", flush=True)

    # 构建 vLLM 启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor-parallel_size),
        "--dtype", args.dtype,
        "--quantization", args.quantization,
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    print(f"[start_teacher_vllm] 执行命令:", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"[start_teacher_vllm] 按 Ctrl+C 停止服务", flush=True)
    print(flush=True)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[start_teacher_vllm] 服务已停止", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] vLLM 启动失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
