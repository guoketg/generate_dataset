"""fetch_product_images.py —— Unsplash 官方 API 商品图采集(PRD 17 5.3.6)。

端点 /search/photos(Demo 档 50 搜索请求/小时,图片下载不限配额)。
特性:429 自动等下一小时窗口;断点续传游标落盘(.fetch_state.json);
已存在文件跳过;下载并发化(--concurrency,默认 8)。图片走 Unsplash License。

Access Key 从环境变量 UNSPLASH_ACCESS_KEY(cloud_training/.env)注入,不入仓库。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

import httpx

SEARCH_URL = "https://api.unsplash.com/search/photos"
QUERIES = {
    "3c": ["smartphone", "laptop", "headphone", "tablet", "camera"],
    "clothing": ["t-shirt", "sneaker", "jacket", "dress", "bag"],
    "home": ["mug", "lamp", "sofa", "knife", "vase"],
    "food": ["coffee", "snack", "fruit", "tea", "chocolate"],
}
PER_WORD = 250
OUT = DATA/"images/products"
STATE = OUT / ".fetch_state.json"


def load_env() -> str:
    """读 .env(不覆盖已有环境变量)"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not key:
        raise SystemExit("UNSPLASH_ACCESS_KEY not set (check cloud_training/.env)")
    return key


async def search_page(client: httpx.AsyncClient, key: str, query: str,
                      page: int, sem: asyncio.Semaphore) -> list[str]:
    async with sem:  # 搜索请求串行(配额敏感:50/h)
        for attempt in range(3):
            r = await client.get(SEARCH_URL, params={
                "query": query, "per_page": 30, "page": page, "client_id": key,
            }, timeout=30)
            if r.status_code == 429:  # 配额尽:等下一小时窗口
                wait = 3600 - (time.time() % 3600) + 5
                print(f"[429] quota exhausted, sleep {wait:.0f}s", flush=True)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return [p["urls"]["regular"] for p in r.json()["results"]]
        return []


async def download(client: httpx.AsyncClient, url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True  # 断点续传:已下载跳过
    try:
        r = await client.get(url, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return True
    except httpx.HTTPError as e:
        print(f"[download] failed {path.name}: {e}", flush=True)
    return False


async def main(concurrency: int = 8):
    key = load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    # 搜索串行(配额敏感:50/h); 下载由 asyncio.gather 并发,无需额外信号量
    search_sem = asyncio.Semaphore(1)
    total = 0
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        for cat, queries in QUERIES.items():
            for q in queries:
                got = state.get(f"{cat}/{q}", 0)
                skipped = 0
                while got < PER_WORD:
                    page = got // 30 + 1
                    urls = await search_page(client, key, q, page, search_sem)
                    if not urls:
                        print(f"[{cat}/{q}] no more results at page {page} "
                              f"(got {got}/{PER_WORD})", flush=True)
                        break
                    tasks = []
                    for j, u in enumerate(urls):
                        if got + j >= PER_WORD:
                            break
                        path = OUT / f"{cat}_{q}_{got + j:04d}.jpg"
                        tasks.append(download(client, u, path))
                        total += 1
                    results = await asyncio.gather(*tasks)
                    skipped += results.count(False)
                    got += sum(1 for r in results if r)
                    state[f"{cat}/{q}"] = got
                    STATE.write_text(json.dumps(state))  # 每页落盘游标
                    print(f"[{cat}/{q}] page {page}: total {got}/{PER_WORD} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    n_img = len(list(OUT.glob("*.jpg")))
    print(f"[fetch_product_images] DONE images={n_img} (target 5000, "
          f"download_fail={skipped}) elapsed={(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    import sys
    conc = int(sys.argv[sys.argv.index("--concurrency") + 1]) \
        if "--concurrency" in sys.argv else 8
    asyncio.run(main(conc))
