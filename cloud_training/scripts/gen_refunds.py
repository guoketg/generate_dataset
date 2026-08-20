"""gen_refunds.py —— 退款状态机测试数据(PRD 17 5.3.4 骨架补全)。

覆盖 09-ticket-generation 退款状态机 5 状态: init/reviewing/approved/refunded/rejected,
每状态 100 条,共 500 行。幂等:refunds.jsonl 存在且 500 行则跳过。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

SEED = 42
REFUND_STATES = ["init", "reviewing", "approved", "refunded", "rejected"]
REFUND_REASONS = {
    "init": "用户发起退款申请，等待审核",
    "reviewing": "客服审核中，需补充瑕疵图",
    "approved": "审核通过，等待退款到账",
    "refunded": "退款已到账，订单关闭",
    "rejected": "退款被拒，原因：商品无瑕疵",
}
OUT = DATA/"refunds.jsonl"


def gen_refunds(n_per_state: int = 100, n_products: int = 5000):
    rng = random.Random(SEED)
    refunds = []
    rid = 1
    for state in REFUND_STATES:
        for _ in range(n_per_state):
            pid = rng.randint(1, n_products)
            refunds.append({
                "refund_id": f"RF{rid:08d}",
                "order_id": f"ORD{pid:08d}",
                "product_id": pid,
                "state": state,
                "reason": REFUND_REASONS[state],
                "amount": round(rng.uniform(20, 2000), 2),
                "created_at": f"2026-08-{rng.randint(1, 18):02d} "
                              f"{rng.randint(8, 22):02d}:00:00",
            })
            rid += 1
    return refunds


def main():
    if OUT.exists() and sum(1 for _ in open(OUT, encoding="utf-8")) >= 500:
        print("[gen_refunds] exists, skip (idempotent)", flush=True)
        return
    t0 = time.time()
    refunds = gen_refunds()
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in refunds)
    print(f"[gen_refunds] DONE {len(refunds)} refunds "
          f"(5 states x 100) elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
