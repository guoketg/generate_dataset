"""gen_logistics.py —— 物流轨迹生成(PRD 17 5.3.3 骨架补全)。

读 data/products.jsonl,每 SKU 一单(order_id = ORD{product_id:08d}),
4-6 轨迹点覆盖 5 状态(shipped/in_transit/delivering/signed/rejected)。
幂等:logistics.jsonl 存在且 5000 行则跳过。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

SEED = 42
STATUSES = ["shipped", "in_transit", "delivering", "signed", "rejected"]
STATUS_CN = {
    "shipped": "已发货", "in_transit": "运输中", "delivering": "派送中",
    "signed": "已签收", "rejected": "已拒收",
}
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
OUT = DATA/"logistics.jsonl"


def gen_track(order_id: str, product_id: int, rng: random.Random) -> dict:
    n_points = rng.randint(4, 6)
    # 前段单调推进 + 终态签收/拒收(骨架语义)
    status_seq = STATUSES[: n_points - 1] + [rng.choice(["signed", "rejected"])]
    route = rng.sample(CITIES, min(n_points, len(CITIES)))  # 路径不重复城市
    trajectory = []
    for i, st in enumerate(status_seq):
        trajectory.append({
            "ts": f"2026-08-{10 + i:02d} {10 + i:02d}:30:00",
            "location": route[i % len(route)],
            "action": st,
            "action_cn": STATUS_CN[st],
        })
    return {
        "order_id": order_id,
        "product_id": product_id,
        "status": status_seq[-1],
        "status_cn": STATUS_CN[status_seq[-1]],
        "trajectory": trajectory,
    }


def main():
    rng = random.Random(SEED)
    if OUT.exists() and sum(1 for _ in open(OUT, encoding="utf-8")) >= 5000:
        print("[gen_logistics] exists, skip (idempotent)", flush=True)
        return
    t0 = time.time()
    tracks = []
    with open("data/products.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            tracks.append(gen_track(f"ORD{p['product_id']:08d}", p["product_id"], rng))
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in tracks)
    dist = {s: 0 for s in STATUSES}
    for t in tracks:
        dist[t["status"]] += 1
    print(f"[gen_logistics] DONE {len(tracks)} tracks, status_dist={dist} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
