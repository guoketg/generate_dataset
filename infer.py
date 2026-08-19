"""
Qwen3.8-27B GGUF 本地推理脚本（不使用 vLLM）。
依赖：llama-cpp-python（CUDA 后端），安装于本仓库 .venv。

用法：
    . .venv/bin/activate
    python infer.py --model models/Qwen3.8-27B-UD-Q4_K_XL.gguf

交互模式：
    python infer.py --model <gguf> --interactive

单轮测试：
    python infer.py --model <gguf> --prompt "你是谁？"
"""
import argparse
import sys

from llama_cpp import Llama


def build_model(model_path: str, n_gpu_layers: int, n_ctx: int, n_threads: int, verbose: bool):
    # 注意：llama-cpp-python 0.3.35 的 Llama 构造与 create_chat_completion
    # 均不接收 chat_template / chat_template_kwargs / reasoning_effort 参数。
    # Qwen3.8 的 GGUF 自带 tokenizer.chat_template，会按内置逻辑默认启用
    # 混合思考（输出 <think>...</think> 段落）。本版本 API 未暴露开关，
    # 思考强度由模型默认行为决定。
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,      # -1 = 全部层卸载到 GPU
        n_ctx=n_ctx,                    # 上下文窗口（Qwen3.8 原生 256K，保守用 8192）
        n_threads=n_threads,
        n_batch=512,
        verbose=verbose,
    )
    return llm


def chat_once(llm: Llama, messages, temperature, top_p, top_k, max_tokens, reasoning_effort=None):
    # reasoning_effort 在本版本暂未暴露到 API，仅作占位保留。
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-27B GGUF 推理（llama-cpp-python）")
    ap.add_argument("--model", required=True, help="GGUF 模型路径")
    ap.add_argument("--n_gpu_layers", type=int, default=-1, help="卸载到 GPU 的层数，-1 全卸载")
    ap.add_argument("--n_ctx", type=int, default=8192, help="上下文长度")
    ap.add_argument("--n_threads", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--reasoning_effort", default="medium",
                    help="思考深度: xhigh/high/medium/low/none")
    ap.add_argument("--prompt", default=None, help="单轮推理文本（非交互模式）")
    ap.add_argument("--interactive", action="store_true", help="进入交互对话")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # chat 模板与思考模式在 chat_once 中通过 chat_template_kwargs 传入
    llm = build_model(args.model, args.n_gpu_layers, args.n_ctx, args.n_threads, args.verbose)

    if args.prompt and not args.interactive:
        messages = [{"role": "user", "content": args.prompt}]
        msg = chat_once(llm, messages, args.temperature, args.top_p, args.top_k,
                        args.max_tokens, args.reasoning_effort)
        print("\n[assistant]:", msg.get("content", ""))
        return

    # 交互模式
    print("Qwen3.8-27B 交互对话（输入 exit/quit 退出，/clear 清空历史）")
    history = []
    while True:
        try:
            user = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "/clear":
            history = []
            print("[history cleared]")
            continue
        history.append({"role": "user", "content": user})
        msg = chat_once(llm, history, args.temperature, args.top_p, args.top_k,
                        args.max_tokens, args.reasoning_effort)
        content = msg.get("content", "")
        history.append({"role": "assistant", "content": content})
        print("\nassistant>", content)


if __name__ == "__main__":
    main()
