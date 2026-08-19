"""
下载 unsloth/Qwen3.8-27B-GGUF 的指定量化版本（国内 HF 镜像）。
依赖：huggingface_hub（已装于 .venv）。

用法：
    . .venv/bin/activate
    export HF_ENDPOINT=https://hf-mirror.com
    python download_model.py --quant UD-Q4_K_XL

可选 quant 示例：UD-Q4_K_XL(推荐,~18GB) / UD-Q3_K_XL(~15GB) / UD-Q6_K / UD-IQ2_XXS
"""
import os
import sys
import argparse

from huggingface_hub import snapshot_download, hf_hub_download

REPO_ID = "unsloth/Qwen3.8-27B-GGUF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="UD-Q4_K_XL",
                    help="量化等级，匹配文件名中的关键字")
    ap.add_argument("--local_dir", default="models")
    args = ap.parse_args()

    # 强制使用国内镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print(f"[info] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"[info] 下载仓库 {REPO_ID} 中匹配 *{args.quant}* 的 GGUF 文件...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"Qwen3.8-27B-{args.quant}.gguf",
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[done] 模型已保存到: {path}")


if __name__ == "__main__":
    main()
