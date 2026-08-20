"""gen_refunds.py —— 退款状态机测试数据(PRD 17 5.3.4)。

覆盖 09-ticket-generation.md 退款状态机 5 种状态,每状态 100 条。
纯规则生成,不依赖 GPU/LLM。幂等:已存在则跳过。
输出: data/refunds.jsonl (500 行)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"
OUT = _DATA / "refunds.jsonl"
SEED = 42

REFUND_STATES = ["init", "reviewing", "approved", "refunded", "rejected"]
REFUND_REASONS = {
    "init": "用户发起退款申请，等待审核",
    "reviewing": "客服审核中，需补充瑕疵图",
    "approved": "审核通过，等待退款到账",
    "refunded": "退款已到账，订单关闭",
    "rejected": "退款被拒，原因：商品无瑕疵",
}
N_PER_STATE = 100


def main():
    if OUT.exists() and sum(1 for _ in open(OUT, encoding="utf-8")) >= len(REFUND_STATES) * N_PER_STATE:
        print(f"[gen_refunds] skip, already exists", flush=True)
        return
    rng = random.Random(SEED)
    refunds = []
    rid = 1
    for state in REFUND_STATES:
        for _ in range(N_PER_STATE):
            pid = rng.randint(1, 5000)
            refunds.append({
                "refund_id": f"RF{rid:08d}",
                "order_id": f"ORD{pid:08d}",
                "product_id": pid,
                "state": state,
                "reason": REFUND_REASONS[state],
                "amount": round(rng.uniform(20, 2000), 2),
                "created_at": f"2026-08-{rng.randint(1, 18):02d} {rng.randint(8, 22):02d}:00:00",
            })
            rid += 1
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in refunds)
    print(f"[gen_refunds] DONE rows={len(refunds)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
