"""fetch_amazon_seeds.py —— Amazon Reviews 2023 元数据流式采样(PRD 17 5.3.2 种子路径)。

流式下载 4 个类目元数据(.jsonl.gz,不全量落盘),每类目水库抽样 ~1300 条有效记录,
真实品牌词替换为虚构品牌表,价格按类目 1%/99% 分位裁剪,输出 data/seeds/{cat}.jsonl。

源: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/
类目映射: Electronics→3c / Clothing_Shoes_and_Jewelry→clothing
          / Home_and_Kitchen→home / Grocery_and_Gourmet_Food→food

幂等:输出文件已存在且行数 >= 阈值时跳过。
"""
from __future__ import annotations

import gzip
import io
import json
import random
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

import httpx

SEED = 42
BASE = ("https://mcauleylab.ucsd.edu/public_datasets/data/"
        "amazon_2023/raw/meta_categories/")
CATEGORIES = {
    "3c": "meta_Electronics.jsonl.gz",
    "clothing": "meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
    "home": "meta_Home_and_Kitchen.jsonl.gz",
    "food": "meta_Grocery_and_Gourmet_Food.jsonl.gz",
}
SAMPLE_PER_CAT = 1300          # 每类目抽样目标(4% 冗余供分位裁剪)
MIN_KEEP = 1250                # 裁剪后最低保留(低于此则告警不失败)
BRANDS = {  # PRD 17 5.3.2 虚构品牌表
    "3c": ["NovaTech", "Pulse", "OrbitX", "VertexQ", "LumenA"],
    "clothing": ["Threadline", "VogueStep", "AuroraWear", "CobaltBay", "MapleCo"],
    "home": ["HearthHome", "Lumio", "CedarWorks", "PaleMoon", "TideStudio"],
    "food": ["BeanVista", "SnackHive", "OrchardGold", "SteepLeaf", "CocoaRidge"],
}
# 标题首词(真实品牌候选)→ 虚构品牌 的稳定映射
REAL2FAKE: dict[str, str] = {}

PRICE_RE = re.compile(r"^\$?([0-9][0-9,]*)(?:\.([0-9]{1,2}))?$")
TITLE_BRAND_RE = re.compile(r"^([A-Za-z][A-Za-z0-9&'’.\-]{0,29})\s")


def parse_price(raw) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    m = PRICE_RE.match(str(raw).strip())
    if not m:
        return None
    val = float(m.group(1).replace(",", "") + "." + (m.group(2) or "0"))
    return val if 0.5 <= val <= 100000 else None


def map_brand(title: str, cat: str, rng: random.Random) -> tuple[str, str]:
    """标题首词视为真实品牌 → 稳定映射到该类目虚构品牌;返回 (新标题, 虚构品牌)"""
    m = TITLE_BRAND_RE.match(title)
    if not m:
        brand = rng.choice(BRANDS[cat])
        return title, brand
    real = m.group(1)
    if real not in REAL2FAKE:
        REAL2FAKE[real] = BRANDS[cat][len(REAL2FAKE) % len(BRANDS[cat])]
    fake = REAL2FAKE[real]
    return title[: m.start()] + fake + title[m.end():], fake


def iter_records(cat: str):
    """流式下载 + 逐行解压解析,不落盘全量"""
    url = BASE + CATEGORIES[cat]
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
        resp.raise_for_status()
        gz = gzip.GzipFile(fileobj=io.BufferedReader(resp.raw))
        for i, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8", errors="ignore")):
            yield i, line


def sample_category(cat: str, out_dir: Path) -> list[dict]:
    rng = random.Random(SEED + hash(cat) % 10000)
    pool: list[dict] = []          # 水库抽样
    all_prices: list[float] = []   # 全量价格(分位数)
    n_seen = n_valid = 0
    t0 = time.time()
    for lineno, line in iter_records(cat):
        n_seen += 1
        if n_seen % 200000 == 0:
            print(f"[{cat}] seen={n_seen} valid={n_valid} pool={len(pool)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"[{cat}] skip bad json line {lineno}", flush=True)
            continue
        title = rec.get("title")
        price = parse_price(rec.get("price"))
        if not title or len(str(title).strip()) < 8 or price is None:
            continue
        title = " ".join(str(title).split())  # 归一化空白
        if len(title) > 120:
            title = title[:120]
        n_valid += 1
        all_prices.append(price)
        item = {"title": title, "price": price, "category": cat}
        if len(pool) < SAMPLE_PER_CAT:
            pool.append(item)
        else:
            j = rng.randint(0, n_valid - 1)
            if j < SAMPLE_PER_CAT:
                pool[j] = item
    # 分位裁剪(1% / 99%)
    all_prices.sort()
    if all_prices:
        lo = all_prices[int(len(all_prices) * 0.01)]
        hi = all_prices[min(int(len(all_prices) * 0.99), len(all_prices) - 1)]
    else:
        lo, hi = 0.0, float("inf")
    kept = [p for p in pool if lo <= p["price"] <= hi]
    # 品牌替换 + 近似去重(去品牌归一化)
    seen_norm = set()
    out: list[dict] = []
    for item in kept:
        new_title, brand = map_brand(item["title"], cat, rng)
        norm = re.sub(r"\s+", "", new_title.lower())[:60]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        out.append({"title": new_title, "brand": brand,
                    "price": round(item["price"], 2), "category": cat})
    # 落盘
    path = out_dir / f"{cat}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(f"[{cat}] DONE seen={n_seen} valid={n_valid} kept={len(out)} "
          f"price=[{lo:.1f},{hi:.1f}] -> {path} ({time.time()-t0:.0f}s)", flush=True)
    if len(out) < MIN_KEEP:
        print(f"[{cat}] WARN: kept {len(out)} < {MIN_KEEP}, "
              f"gen_products 将用 pure 模式兜底", flush=True)
    return out


def main():
    random.seed(SEED)
    out_dir = DATA/"seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for cat in CATEGORIES:
        path = out_dir / f"{cat}.jsonl"
        if path.exists() and sum(1 for _ in open(path, encoding="utf-8")) >= MIN_KEEP:
            print(f"[{cat}] exists, skip (idempotent)", flush=True)
            continue
        try:
            total += len(sample_category(cat, out_dir))
        except Exception as e:  # noqa: BLE001 —— 单类目失败不阻断其余
            print(f"[{cat}] FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"[fetch_amazon_seeds] total seeds = {total}", flush=True)


if __name__ == "__main__":
    main()
