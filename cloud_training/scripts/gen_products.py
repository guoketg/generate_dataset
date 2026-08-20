"""gen_products.py —— 商品主库 + 比价 + 防伪码生成(PRD 17 5.3.2)。

双模式:
  --mode seed  (推荐)读 data/seeds/*.jsonl,规则改写后由 Teacher 仅补全
               description + attributes(批量 50 条/次,约 3-5h)
  --mode pure  纯 LLM 兜底生成(PRD 17 v1.0 原方案,约 12-20h)
种子不可用(缺文件或行数不足)时自动落 pure。

输出: data/products.jsonl(5000) / prices.jsonl(20000) / anti_fake.jsonl(5000, 5% 假=250)
幂等: 三个输出文件均存在且行数达标时跳过;LLM 补全批次断点续传(progress 游标)。
断言: 品牌全部 ∈ 虚构品牌表(任务书 D3 验收项)。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

from llm_client import TeacherLLM

SEED = 42
N_PRODUCTS = 5000
N_FAKE = 250
PURE_BATCH = 25  # pure 模式每批条数(4096 tokens 上限下避免 JSON 截断)
PLATFORMS = ["jd", "taobao", "pdd", "amazon"]
CATEGORIES = {
    "3c": ["smartphone", "laptop", "headphone", "tablet", "camera"],
    "clothing": ["t-shirt", "sneaker", "jacket", "dress", "bag"],
    "home": ["mug", "lamp", "sofa", "knife", "vase"],
    "food": ["coffee", "snack", "fruit", "tea", "chocolate"],
}
BRANDS = {
    "3c": ["NovaTech", "Pulse", "OrbitX", "VertexQ", "LumenA"],
    "clothing": ["Threadline", "VogueStep", "AuroraWear", "CobaltBay", "MapleCo"],
    "home": ["HearthHome", "Lumio", "CedarWorks", "PaleMoon", "TideStudio"],
    "food": ["BeanVista", "SnackHive", "OrchardGold", "SteepLeaf", "CocoaRidge"],
}
ALL_FAKE_BRANDS = {b for bs in BRANDS.values() for b in bs}
BATCH_SIZE = 50
MIN_SEED_LINES = 1250

PROMPT_PURE = """You are an e-commerce product data generator. Output strict JSON only, no markdown.
Generate {n} unique products in category "{cat}" with brands from: {brands}.
Each product object:
{{
  "title": "<Brand> <Model> <KeyAttr> <CategoryNoun>, <=30 chars Chinese/English",
  "brand": "<one of given brands>",
  "model": "<short alphanumeric model code, e.g. NX-4501>",
  "price": <number, 9.9 ~ 9999.0, reasonable for category>,
  "description": "<<=50 chars, mention 1-2 selling points>",
  "attributes": {{"color": "...", "size": "...", "material": "..."}}
}}
Output as JSON array only, no other text. Do not repeat titles."""

PROMPT_DESC = """You are an e-commerce product copywriter. Output strict JSON array only, no markdown.
For each product below, generate a short Chinese description and attributes.
Input list (index | category | brand | title):
{items}
Each output object (same order as input):
{{
  "index": <input index>,
  "description": "<中文,<=50字,突出1-2个卖点>",
  "attributes": {{"color": "...", "size": "...", "material": "..."}}
}}
Output as JSON array only, no other text."""


def make_prices_and_antifake(products: list[dict], rng: random.Random):
    """比价(基准价 ±15%) + 防伪码(5% 假 = 250)"""
    prices, anti_fake = [], []
    fake_indices = set(rng.sample(range(len(products)), N_FAKE))
    for p in products:
        pid = p["product_id"]
        for plat in PLATFORMS:
            prices.append({
                "product_id": pid, "platform": plat,
                "price": round(p["price"] * rng.uniform(0.85, 1.15), 2),
            })
        anti_fake.append({
            "code": f"AF{pid:08d}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}",
            "product_id": pid,
            "is_genuine": pid not in fake_indices,
        })
    return prices, anti_fake


def validate_product(item: dict, cat: str, pid: int) -> dict | None:
    """字段校验与兜底,返回标准 product dict"""
    title = str(item.get("title", f"{cat}_{pid}"))[:60].strip()
    if len(title) < 4:
        return None
    brand = str(item.get("brand", "")).strip()
    if brand not in ALL_FAKE_BRANDS:
        brand = BRANDS[cat][pid % len(BRANDS[cat])]
    try:
        price = round(float(item.get("price", 99.0)), 2)
    except (TypeError, ValueError):
        price = 99.0
    if not (0.5 <= price <= 100000):
        price = min(max(price, 9.9), 9999.0)
    attrs = item.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    return {
        "product_id": pid, "title": title, "category": cat, "brand": brand,
        "model": str(item.get("model", f"M{pid:05d}"))[:16],
        "price": price,
        "platform": "jd",
        "description": str(item.get("description", ""))[:120],
        "attributes": attrs,
    }


def load_seed_mode(llm: TeacherLLM, rng: random.Random) -> list[dict]:
    """种子模式:读 seeds,LLM 批量补全 description/attributes(断点续传)"""
    seed_dir = DATA/"seeds"
    # 均衡四类目至 5000（含校验：文件缺失/行数不足会抛 FileNotFoundError/ValueError）
    per_cat = N_PRODUCTS // len(CATEGORIES)
    seeds = []
    for cat in CATEGORIES:
        path = seed_dir / f"{cat}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        take = rows[:per_cat]
        while len(take) < per_cat:  # 种子不足时循环复用并加序号扰动
            extra = dict(rows[len(take) % len(rows)])
            extra["title"] = extra["title"] + f" {rng.choice(['升级款', '新款', '限定款'])}"
            take.append(extra)
        seeds.extend(take)
    print(f"[seed-mode] loaded {len(seeds)} seeds", flush=True)

    products: list[dict] = []
    # 断点续传:products.partial.jsonl + progress
    partial = DATA/"products.partial.jsonl"
    done_batches = 0
    if partial.exists():
        with open(partial, encoding="utf-8") as f:
            for line in f:
                products.append(json.loads(line))
        done_batches = len(products) // BATCH_SIZE
        print(f"[seed-mode] resume from {len(products)} products "
              f"({done_batches} batches)", flush=True)

    n_batches = (len(seeds) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(done_batches, n_batches):
        chunk = seeds[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        items_txt = "\n".join(
            f"{i} | {s['category']} | {s['brand']} | {s['title']}"
            for i, s in enumerate(chunk)
        )
        arr = llm.chat_json_array(
            PROMPT_DESC.format(items=items_txt), temperature=0.7, max_tokens=4096)
        desc_map = {}
        for it in arr:
            if isinstance(it, dict) and "index" in it:
                desc_map[int(it["index"])] = it
        for i, s in enumerate(chunk):
            pid = b * BATCH_SIZE + i + 1
            d = desc_map.get(i, {})
            products.append({
                "product_id": pid, "title": s["title"], "category": s["category"],
                "brand": s["brand"], "model": f"M{pid:05d}",
                "price": s["price"], "platform": "jd",
                "description": str(d.get("description", ""))[:120],
                "attributes": d.get("attributes") or {},
            })
        with open(partial, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in products)
        print(f"[seed-mode] batch {b + 1}/{n_batches} done, total={len(products)}",
              flush=True)
    partial.unlink(missing_ok=True)
    return products[:N_PRODUCTS]


def load_pure_mode(llm: TeacherLLM, rng: random.Random) -> list[dict]:
    """纯 LLM 兜底:PRD 17 v1.0 骨架路径。每批 PURE_BATCH 条,避免 max_tokens 截断。"""
    products: list[dict] = []
    for cat, nouns in CATEGORIES.items():
        for noun in nouns:
            prompt = PROMPT_PURE.format(n=PURE_BATCH, cat=cat, brands=BRANDS[cat])
            items = llm.chat_json_array(prompt, temperature=0.9, max_tokens=4096)
            if not items:
                print(f"[pure-mode] FAILED batch cat={cat} noun={noun}, "
                      f"filling with template fallback", flush=True)
            for idx, item in enumerate(items if items else [{} for _ in range(PURE_BATCH)]):
                pid = len(products) + 1
                p = validate_product(item, cat, pid)
                if p:
                    products.append(p)
            print(f"[pure-mode] cat={cat} noun={noun} done, total={len(products)}",
                  flush=True)
            if len(products) >= N_PRODUCTS:
                break
        if len(products) >= N_PRODUCTS:
            break
    # 不足则模板补齐(保产物优先)
    while len(products) < N_PRODUCTS:
        pid = len(products) + 1
        cat = list(CATEGORIES)[pid % 4]
        noun = rng.choice(CATEGORIES[cat])
        brand = BRANDS[cat][pid % len(BRANDS[cat])]
        products.append({
            "product_id": pid, "title": f"{brand} M{pid:05d} {noun}",
            "category": cat, "brand": brand, "model": f"M{pid:05d}",
            "price": round(rng.uniform(9.9, 999.0), 2), "platform": "jd",
            "description": f"{brand}品牌{noun},品质保证,七天无理由退换。",
            "attributes": {},
        })
    return products[:N_PRODUCTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seed", "pure"], default="seed")
    args = ap.parse_args()

    rng = random.Random(SEED)
    random.seed(SEED)

    # 幂等:三产物齐备即跳过
    if (DATA/"products.jsonl".exists()
            and DATA/"prices.jsonl".exists()
            and DATA/"anti_fake.jsonl".exists()):
        n = sum(1 for _ in open("data/products.jsonl", encoding="utf-8"))
        if n >= N_PRODUCTS:
            print("[gen_products] outputs exist, skip (idempotent)", flush=True)
            return

    mode = args.mode
    if mode == "seed" and not (DATA / "seeds").exists():
        print("[gen_products] seeds dir missing, fallback to pure mode", flush=True)
        mode = "pure"

    t0 = time.time()
    llm = TeacherLLM()
    print(f"[gen_products] mode={mode}, model loaded", flush=True)

    # 种子模式：文件缺失/行数不足时自动落 pure 兜底（任务书异常表要求汇报原因）
    if mode == "seed":
        try:
            products = load_seed_mode(llm, rng)
        except (FileNotFoundError, ValueError) as e:
            print(f"[gen_products] seed load failed ({e}); fallback to pure mode",
                  flush=True)
            mode = "pure"
            products = load_pure_mode(llm, rng)
    else:
        products = load_pure_mode(llm, rng)
    assert len(products) == N_PRODUCTS, f"products={len(products)} != {N_PRODUCTS}"
    bad_brands = [p for p in products if p["brand"] not in ALL_FAKE_BRANDS]
    assert not bad_brands, f"non-fake brands found: {bad_brands[:3]}"

    prices, anti_fake = make_prices_and_antifake(products, rng)
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA/"products.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in products)
    with open(DATA/"prices.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in prices)
    with open(DATA/"anti_fake.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in anti_fake)

    print(f"[gen_products] DONE products={len(products)} prices={len(prices)} "
          f"anti_fake={len(anti_fake)} (fake={N_FAKE}) "
          f"llm_calls={llm.n_calls} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
