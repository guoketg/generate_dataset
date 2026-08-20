"""fill_missing_desc.py —— 仅补全 products.jsonl 中 description 为空的条目。

走已运行的 teacher HTTP 服务(127.0.0.1:8001),不依赖 llama_cpp 本地加载。
直接 httpx 调用 /v1/chat/completions,绕开 llm_client 的 Llama 初始化(那会强制
import llama_cpp 并加载 17GB GGUF)。

用法:
  python3 scripts/fill_missing_desc.py --base-url http://127.0.0.1:8001
幂等:再次运行只补仍为空者。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

_DATA = Path(__file__).resolve().parents[1] / "data"
PRODUCTS = _DATA / "products.jsonl"


def chat_json_array(base_url: str, prompt: str, temperature: float, max_tokens: int):
    """调用 teacher 服务取 JSON 数组,带容错解析。"""
    r = httpx.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json={
            "model": "teacher",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=600.0,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # 去除可能的 <think> 段与 markdown 包裹
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(content[start:end + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(PRODUCTS, encoding="utf-8")]
    missing = [r for r in rows if not (r.get("description") or "").strip()]
    print(f"[fill_desc] total={len(rows)} empty_desc={len(missing)}", flush=True)
    if not missing:
        print("[fill_desc] nothing to fill, done", flush=True)
        return

    for i in range(0, len(missing), args.batch):
        batch = missing[i:i + args.batch]
        prompt = "Write concise product descriptions. Output strict JSON array only.\n"
        for r in batch:
            prompt += (
                f"- product_id={r['product_id']} title={r['title']} "
                f"brand={r.get('brand','')} category={r['category']}\n"
            )
        prompt += ('Return JSON: [{"product_id": <int>, "description": "<Chinese 30-60 chars, 1-2 selling points>"}, ...]')

        try:
            arr = chat_json_array(args.base_url, prompt, temperature=0.7, max_tokens=2048)
        except Exception as e:
            print(f"[fill_desc] batch {i // args.batch + 1} ERROR: {e}", flush=True)
            continue
        by_pid = {item["product_id"]: item["description"]
                  for item in arr if isinstance(item, dict) and "product_id" in item}
        filled = 0
        for r in batch:
            d = by_pid.get(r["product_id"])
            if d and str(d).strip():
                r["description"] = str(d).strip()
                filled += 1
        print(f"[fill_desc] batch {i // args.batch + 1}: filled {filled}/{len(batch)}", flush=True)

    with open(PRODUCTS, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    still = sum(1 for r in rows if not (r.get("description") or "").strip())
    print(f"[fill_desc] DONE. remaining empty_desc={still}", flush=True)


if __name__ == "__main__":
    main()
