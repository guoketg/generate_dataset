"""gen_logistics.py —— 为每 SKU 生成 1 条物流轨迹(PRD 17 5.3.3)。

纯规则生成,不依赖 GPU/LLM。幂等:已存在则跳过。
输出: data/logistics.jsonl (5000 行)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"
OUT = _DATA / "logistics.jsonl"
SEED = 42

STATUSES = ["shipped", "in_transit", "delivering", "signed", "rejected"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]


def gen_track(order_id: str, product_id: int) -> dict:
    rng = random.Random(product_id)  # 按 product_id 固定,可复现
    n_points = rng.randint(4, 6)
    status_seq = STATUSES[: n_points - 1] + [rng.choice(["signed", "rejected"])]
    trajectory = []
    for i, st in enumerate(status_seq):
        trajectory.append({
            "ts": f"2026-08-{10 + i:02d} {10 + i:02d}:30:00",
            "location": rng.choice(CITIES),
            "action": st,
        })
    return {
        "order_id": order_id,
        "product_id": product_id,
        "status": status_seq[-1],
        "trajectory": trajectory,
    }


def main():
    products = [json.loads(l) for l in open(_DATA / "products.jsonl", encoding="utf-8")]
    if OUT.exists() and sum(1 for _ in open(OUT, encoding="utf-8")) >= len(products):
        print(f"[gen_logistics] skip, already {sum(1 for _ in open(OUT, encoding='utf-8'))} rows", flush=True)
        return
    random.seed(SEED)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in products:
            t = gen_track(f"ORD{p['product_id']:08d}", p["product_id"])
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[gen_logistics] DONE rows={len(products)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
