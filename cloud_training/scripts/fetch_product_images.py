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
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

import httpx

# 随机 UA 池,降低 CDN 指纹关联导致的限流/封 IP 风险
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]
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
            try:
                r = await client.get(SEARCH_URL, params={
                    "query": query, "per_page": 30, "page": page, "client_id": key,
                }, timeout=30, headers={"User-Agent": random.choice(_USER_AGENTS)})
            except httpx.HTTPError as e:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                print(f"[search] {query} p{page} http error {e}, "
                      f"retry in {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
                continue
            if r.status_code == 429 or r.status_code == 403:
                # 配额耗尽或临时封禁:等至下一整点窗口再续
                wait = 3600 - (time.time() % 3600) + 5
                print(f"[search] {query} p{page} http {r.status_code} "
                      f"(quota/blocked), sleep {wait:.0f}s to next hour",
                      flush=True)
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 500:  # 服务端错误:短退避重试
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                print(f"[search] {query} p{page} http {r.status_code}, "
                      f"retry in {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return [p["urls"]["regular"] for p in r.json()["results"]]
        return []


async def download(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                   url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True  # 断点续传:已下载跳过
    # 随机延时打散请求节奏,降低被 CDN 风控的概率
    await asyncio.sleep(random.uniform(0.1, 0.4))
    async with sem:  # 限制下载并发
        for attempt in range(4):
            try:
                headers = {"User-Agent": random.choice(_USER_AGENTS)}
                r = await client.get(url, timeout=60, headers=headers)
                if r.status_code == 200:
                    path.write_bytes(r.content)
                    return True
                # 429/5xx(含 503 Service Unavailable 封 IP 信号):指数退避后重试
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = (2 ** attempt) * 3 + random.uniform(0, 2)
                    print(f"[download] {path.name} http {r.status_code}, "
                          f"retry in {wait:.1f}s (attempt {attempt+1}/4)", flush=True)
                    await asyncio.sleep(wait)
                    continue
                print(f"[download] {path.name} unexpected http {r.status_code}",
                      flush=True)
                return False
            except httpx.HTTPError as e:
                wait = (2 ** attempt) * 3 + random.uniform(0, 2)
                print(f"[download] failed {path.name}: {e}, retry in {wait:.1f}s",
                      flush=True)
                await asyncio.sleep(wait)
    return False


async def main(concurrency: int = 3):
    key = load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    # 搜索串行(配额敏感:50/h); 下载由并发信号量限流
    search_sem = asyncio.Semaphore(1)
    dl_sem = asyncio.Semaphore(concurrency)  # 限制下载并发,降低风控
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
                        tasks.append(download(client, dl_sem, u, path))
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
        if "--concurrency" in sys.argv else 3
    asyncio.run(main(conc))
