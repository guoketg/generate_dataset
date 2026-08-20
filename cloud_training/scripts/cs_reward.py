"""cs_reward.py —— 规则判分 Reward(PRD 18 5.4)。

R = 0.2*R_format + 0.5*R_answer + 0.2*R_process + 0.1*R_cost

双入口:
1. CLI 离线判分: python cs_reward.py --traj data/xxx.jsonl
2. ms-swift external: --reward_funcs external:cs_reward → reward_func()

轨迹输入格式(JSONL 行):
  {"route": ..., "gold": {...}, "n_ref": int, "messages": [...ms-swift 轨迹...],
   "user_claimed_refund": bool(可选)}

单测: python cs_reward.py --selftest  (每路由 >= 20 case,D0 验收项)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unittest

W_FORMAT, W_ANSWER, W_PROCESS, W_COST = 0.2, 0.5, 0.2, 0.1
MAX_TOOL_LOOP = 5

TOOL_NAMES = {"ocr", "vl_describe", "image_search", "text_search",
              "price_compare", "authenticity_check", "query_logistics",
              "query_refund", "create_refund_ticket", "ask_user",
              "transfer_to_human"}
# 简化 Schema(required 字段 + 类型)
TOOL_SCHEMA = {
    "ocr": {"required": {"image_ref": str},
            "optional": {"focus": str}},
    "vl_describe": {"required": {"image_ref": str, "question": str}},
    "image_search": {"required": {"image_ref": str},
                     "optional": {"top_k": int, "category": str}},
    "text_search": {"required": {"query": str},
                    "optional": {"scope": str, "top_k": int}},
    "price_compare": {"required": {"product_id": int},
                      "optional": {"platforms": list}},
    "authenticity_check": {"required": {"code": str}},
    "query_logistics": {"required": {"order_id": str}},
    "query_refund": {"required": {}, "optional": {"order_id": str, "refund_id": str}},
    "create_refund_ticket": {"required": {"order_id": str, "reason": str},
                             "optional": {"flaw_description": str,
                                          "amount": (int, float)}},
    "ask_user": {"required": {"question": str},
                 "optional": {"expect": str}},
    "transfer_to_human": {"required": {"reason": str},
                          "optional": {"summary": str}},
}

PLATFORM_CN = {"jd": "京东", "taobao": "淘宝", "pdd": "拼多多", "amazon": "亚马逊"}
STATUS_CN = {"shipped": "已发货", "in_transit": "运输中", "delivering": "派送中",
             "signed": "已签收", "rejected": "已拒收"}
REFUND_STATE_CN = {"init": "已发起待审核", "reviewing": "审核中",
                   "approved": "审核通过待到账", "refunded": "已退款",
                   "rejected": "已拒绝"}
GENUINE_WORDS = ["正品", "真品", "是真", " genuine", "官方正品", "真货"]
FAKE_WORDS = ["假货", "仿冒", "是假", "fake", "山寨", "不是正品", "非正品"]
NOT_FOUND_WORDS = ["不存在", "查无此单", "未登记", "没有找到", "无记录",
                   "未查到", "没有查到", "无法查询到"]
REFUSE_WORDS = ["无法提供", "不能提供", "无法查询他人", "没有权限",
                "无法修改", "不能修改", "抱歉，这超出了", "无法执行", "不能执行"]
ORDER_ID_RE = re.compile(r"ORD\d{8}")
REFUND_ID_RE = re.compile(r"RF\d{8}")
AF_CODE_RE = re.compile(r"AF\d{8}[A-Z]")


# ---------------- 轨迹解析 ----------------
def parse_traj(messages: list) -> dict:
    """解析轨迹:assistant 轮(tool_calls/终答)、tool 观测、轮数与调用数"""
    assistant_turns, tool_obs = [], []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            assistant_turns.append({
                "tool_calls": m.get("tool_calls") or [],
                "content": m.get("content") or "",
            })
        elif role == "tool":
            try:
                obs = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                obs = {"success": False, "error": {"code": "INVALID_FORMAT"}}
            tool_obs.append({"name": m.get("name", ""), "obs": obs})
    final_content = ""
    n_tool_call_rounds = 0
    for t in assistant_turns:
        if t["tool_calls"]:
            n_tool_call_rounds += 1
        else:
            final_content = t["content"]  # 最后一次纯文本终答
    n_calls = sum(len(t["tool_calls"]) for t in assistant_turns)
    called_names = [tc["function"]["name"]
                    for t in assistant_turns for tc in t["tool_calls"]
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict)]
    return {
        "turns": assistant_turns, "final": final_content,
        "tool_obs": tool_obs, "n_calls": n_calls,
        "n_rounds": n_tool_call_rounds, "called_names": called_names,
    }


def valid_tool_call(tc: dict) -> bool:
    """单个 tool_call:JSON/工具名/Schema 校验"""
    if not isinstance(tc, dict):
        return False
    fn = tc.get("function")
    if not isinstance(fn, dict) or fn.get("name") not in TOOL_NAMES:
        return False
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False
    if not isinstance(args, dict):
        return False
    schema = TOOL_SCHEMA[fn["name"]]
    for k, typ in schema.get("required", {}).items():
        if k not in args or not isinstance(args[k], typ) or args[k] == "":
            return False
    for k, typ in schema.get("optional", {}).items():
        if k in args and args[k] is not None and not isinstance(args[k], typ):
            return False
    return True


# ---------------- 四项判分 ----------------
def r_format(parsed: dict) -> float:
    calls = [tc for t in parsed["turns"] for tc in t["tool_calls"]]
    if not calls:
        return 1.0 if parsed["final"].strip() else 0.0  # 无工具直答视为合法
    bad_json_or_name, bad_fields = 0, 0
    for tc in calls:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if (not isinstance(fn, dict) or fn.get("name") not in TOOL_NAMES
                or isinstance(fn.get("arguments"), str)
                and not _is_json(fn.get("arguments"))):
            bad_json_or_name += 1
        elif not valid_tool_call(tc):
            bad_fields += 1
    if bad_json_or_name:
        return 0.0
    if bad_fields:
        return 0.5
    return 1.0


def _is_json(s) -> bool:
    try:
        json.loads(s)
        return True
    except (TypeError, json.JSONDecodeError):
        return False


def r_answer(route: str, gold: dict, parsed: dict) -> float:
    final = parsed["final"]
    obs_strs = [json.dumps(o["obs"], ensure_ascii=False) for o in parsed["tool_obs"]]
    all_text = final + " " + " ".join(obs_strs)
    if route == "adversarial":
        return _r_answer_adversarial(gold, parsed)
    if route == "multi":
        subs = gold.get("subs", [])
        if not subs:
            return 0.0
        return sum(r_answer(s["route"], s.get("gold", s), parsed)
                   for s in subs) / len(subs)
    if route == "same_item":
        return _r_answer_same_item(gold, all_text, final)
    if route == "authenticity":
        return _r_answer_authenticity(gold, final)
    if route == "logistics":
        return _r_answer_logistics(gold, all_text, final)
    if route == "refund_create":
        return _r_answer_refund_create(gold, parsed)
    if route == "refund_track":
        return _r_answer_refund_track(gold, all_text, final)
    if route == "consult":
        return _r_answer_consult(gold, final)
    return 0.0


def _r_answer_same_item(gold, all_text, final) -> float:
    """PRD 18 5.4:终答含正确商品标识 且 最低价平台正确 = 1.0(判终答,非轨迹)"""
    title_hit = False
    title = str(gold.get("title", ""))
    pid = gold.get("product_id")
    if title and title[:8] in final:
        title_hit = True
    if pid is not None and str(pid) in final:
        title_hit = True
    plat = gold.get("lowest_platform", "")
    price = gold.get("lowest_price", 0)
    plat_hit = bool(plat) and PLATFORM_CN.get(plat, plat) in final
    price_hit = any(
        abs(_to_num(m) - price) <= 1.0
        for m in re.findall(r"(\d+(?:\.\d+)?)", final))
    score = 0.0
    if title_hit:
        score += 0.5
    if plat_hit and price_hit:
        score += 0.5
    elif plat_hit or price_hit:
        score += 0.25
    return min(score, 1.0)


def _to_num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return -1e9


def _r_answer_authenticity(gold, final) -> float:
    """防伪判分:考虑否定前缀("不是/非/并非"等)翻转 genuine 陈述。

    - 纯正向 genuine 词(正品/真品/真货)且无非前缀 → 声称真货
    - genuine 词被否定前缀修饰(不是正品)→ 声称假货
    - 显式 fake 词(假货/仿冒/山寨/非正品)→ 声称假货
    """
    is_genuine = bool(gold.get("is_genuine"))
    NEG_PREFIXES = ["不是", "非", "并非", "没有", "未", "不"]
    has_neg = any(p in final for p in NEG_PREFIXES)
    g_pos = any(w in final for w in ["正品", "真品", "是真", " genuine", "官方正品", "真货"])
    f_pos = any(w in final for w in ["假货", "仿冒", "是假", "fake", "山寨",
                                      "不是正品", "非正品", "并非正品"])
    says_genuine = g_pos and not has_neg
    says_fake = f_pos or (g_pos and has_neg)
    if says_genuine and not says_fake:
        return 1.0 if is_genuine else 0.0
    if says_fake and not says_genuine:
        return 0.0 if is_genuine else 1.0
    return 0.0


def _r_answer_logistics(gold, all_text, final) -> float:
    status_cn = STATUS_CN.get(gold.get("status", ""), "")
    score = 0.0
    if status_cn and status_cn in final:
        score += 0.6
    last = gold.get("last_point", {})
    if last:
        city, action = last.get("location", ""), STATUS_CN.get(last.get("action", ""), "")
        if city and city in final and action and action in final:
            score += 0.4
    return score


def _r_answer_refund_create(gold, parsed) -> float:
    created_ok = False
    for o in parsed["tool_obs"]:
        if o["name"] == "create_refund_ticket" and o["obs"].get("success"):
            created_ok = True
    if not created_ok:
        return 0.0  # 应建未建
    final = parsed["final"]
    order_ok = str(gold.get("order_id", "")) in final or gold.get("order_id") in [
        s for s in re.findall(r"ORD\d{8}", json.dumps(parsed["tool_obs"]))]
    rid_ok = any(REFUND_ID_RE.findall(
        json.dumps(o["obs"], ensure_ascii=False)) for o in parsed["tool_obs"])
    return 1.0 if (order_ok or rid_ok) else 0.5


def _r_answer_refund_track(gold, all_text, final) -> float:
    state_cn = REFUND_STATE_CN.get(gold.get("state", ""), "")
    score = 0.0
    if state_cn and state_cn in final:
        score += 0.6
    rid = gold.get("refund_id", "")
    oid = gold.get("order_id", "")
    if (rid and rid in final) or (oid and oid in final):
        score += 0.4
    return score


def _r_answer_consult(gold, final) -> float:
    kws = gold.get("gold_keywords", [])
    if not kws:
        return 0.0
    hits = sum(1 for k in kws if k in final)
    ratio = hits / len(kws)
    return 1.0 if ratio >= 0.5 else 2.0 * ratio


def _r_answer_adversarial(gold, parsed) -> float:
    behavior = gold.get("correct_behavior", "")
    final, names = parsed["final"], parsed["called_names"]
    if behavior in ("ask_user", "clarify"):
        if "ask_user" in names:
            return 1.0
        if any(w in final for w in NOT_FOUND_WORDS):
            return 0.8
        return 0.0
    if behavior == "transfer_to_human":
        return 1.0 if "transfer_to_human" in names else 0.0
    if behavior == "state_not_found":
        if any(w in final for w in NOT_FOUND_WORDS):
            return 1.0
        if "ask_user" in names:
            return 0.8
        return 0.0
    if behavior == "refuse":
        if any(w in final for w in REFUSE_WORDS) or "拒绝" in final:
            return 1.0
        if "transfer_to_human" in names:
            return 0.8
        return 0.0
    return 0.0


def r_process(parsed: dict, traj: dict) -> float:
    score = 1.0
    obs_list = [o["obs"] for o in parsed["tool_obs"]]
    violations = sum(
        1 for o in obs_list
        if not o.get("success") and o.get("error", {}).get("code") == "GROUNDING_VIOLATION")
    score -= 0.25 * violations
    no_evidence = sum(
        1 for o in obs_list
        if not o.get("success") and o.get("error", {}).get("code") == "NO_EVIDENCE")
    score -= 0.5 * no_evidence
    # 无证据建单:「成功」建单且事前无证据(执行器已拦截,防御性检测)→ 0;
    # 试图无证据建单被拒由 NO_EVIDENCE 扣分覆盖,不重复计罚
    user_claimed = bool(traj.get("user_claimed_refund"))
    evidence_seen = user_claimed
    for t in parsed["turns"]:
        for tc in t["tool_calls"]:
            fn = tc.get("function", {})
            if fn.get("name") in ("vl_describe", "ocr"):
                evidence_seen = True
            if fn.get("name") == "create_refund_ticket" and not evidence_seen:
                for o in parsed["tool_obs"]:
                    if (o["name"] == "create_refund_ticket"
                            and o["obs"].get("success")):
                        score = 0.0
    if parsed["n_rounds"] > MAX_TOOL_LOOP:
        score -= 0.2
    # 对抗编造:失败观测后终答给出具体编造结论
    if traj.get("route") == "adversarial":
        has_failed_lookup = any(
            (not o.get("success")) and o.get("error", {}).get("code") in
            ("NOT_FOUND", "NOT_REGISTERED", "NO_MATCH") for o in obs_list)
        fabricated = bool(re.search(r"(已签收|运输中|派送中|已发货|已拒收|是正品|假货)",
                                    parsed["final"])) and has_failed_lookup
        if fabricated:
            score -= 0.5
    return max(score, 0.0)


def r_cost(n_calls: int, n_ref: int) -> float:
    if n_calls <= n_ref:
        return 1.0
    if n_calls <= n_ref + 2:
        return 1.0 - (n_calls - n_ref) / 2.0
    return 0.0


def judge(traj: dict) -> dict:
    """完整判分一条轨迹。traj 含 route/gold/n_ref/messages"""
    parsed = parse_traj(traj.get("messages", []))
    rf = r_format(parsed)
    ra = r_answer(traj.get("route", ""), traj.get("gold", {}), parsed)
    rp = r_process(parsed, traj)
    rc = r_cost(parsed["n_calls"], int(traj.get("n_ref", 1)))
    total = (W_FORMAT * rf + W_ANSWER * ra + W_PROCESS * rp + W_COST * rc)
    return {"R_format": rf, "R_answer": ra, "R_process": rp, "R_cost": rc,
            "R_total": round(total, 4), "n_calls": parsed["n_calls"]}


# ---------------- ms-swift external 入口 ----------------
def reward_func(completions, **kwargs) -> list:
    """ms-swift GRPO 自定义 reward(标准签名 func(completions, **kwargs))。

    completions: list[str/消息列表];kwargs 透传数据集列(gold/n_ref/route/
    user_claimed_refund 与多轮上下文 messages)。
    """
    rewards = []
    golds = kwargs.get("gold") or [None] * len(completions)
    n_refs = kwargs.get("n_ref") or [1] * len(completions)
    routes = kwargs.get("route") or [None] * len(completions)
    claimed = kwargs.get("user_claimed_refund") or [False] * len(completions)
    if not isinstance(golds, list):
        golds = [golds] * len(completions)
    if not isinstance(n_refs, list):
        n_refs = [n_refs] * len(completions)
    if not isinstance(routes, list):
        routes = [routes] * len(completions)
    if not isinstance(claimed, list):
        claimed = [claimed] * len(completions)
    for comp, g, nr, rt, cl in zip(completions, golds, n_refs, routes, claimed):
        if isinstance(comp, str):
            try:
                comp = json.loads(comp)
            except json.JSONDecodeError:
                comp = [{"role": "assistant", "content": comp}]
        traj = {"route": rt, "gold": g or {}, "n_ref": nr,
                "user_claimed_refund": cl, "messages": comp}
        rewards.append(judge(traj)["R_total"])
    return rewards


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", help="轨迹 JSONL(route/gold/n_ref/messages)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        unittest.main(argv=[sys.argv[0]], exit=False)
        return
    if not args.traj:
        ap.error("--traj or --selftest required")
    rows = [json.loads(l) for l in open(args.traj, encoding="utf-8")]
    scores = []
    for i, traj in enumerate(rows):
        try:
            r = judge(traj)
            scores.append(r)
            print(f"[{i}] route={traj.get('route')} R={r['R_total']:.3f} "
                  f"(f={r['R_format']} a={r['R_answer']} p={r['R_process']} "
                  f"c={r['R_cost']} n={r['n_calls']})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] JUDGE FAILED: {type(e).__name__}: {e}", flush=True)
    if scores:
        tot = [s["R_total"] for s in scores]
        zero = sum(1 for t in tot if t == 0)
        print(f"\n[summary] n={len(tot)} mean={sum(tot)/len(tot):.4f} "
              f"zero_reward={zero}/{len(tot)}", flush=True)


# ---------------- 单测(每路由 >= 20 case) ----------------
def _traj(route, gold, n_ref, turns, claimed=False):
    """构造轨迹:turns = [(tool_calls|None, content|None), ...] 交错 tool obs"""
    messages = [{"role": "system", "content": "sys"}]
    for t in turns:
        if isinstance(t, tuple) and t[0] == "tool":
            messages.append({"role": "tool", "name": t[1], "content": json.dumps(
                t[2], ensure_ascii=False)})
        else:
            messages.append(t)
    return {"route": route, "gold": gold, "n_ref": n_ref,
            "user_claimed_refund": claimed, "messages": messages}


def _tc(name, args):
    return {"id": "call_001", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _tc_obj(name, args):
    return {"id": "call_001", "type": "function",
            "function": {"name": name, "arguments": args}}


def _asst(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


class TestSameItem(unittest.TestCase):
    GOLD = {"product_id": 1042, "title": "VogueStep VS-2203 轻量跑步鞋",
            "lowest_platform": "pdd", "lowest_price": 349.5}

    def _t(self, final, n_ref=2, calls=2):
        return _traj("same_item", self.GOLD, n_ref, [
            _asst(tool_calls=[_tc("image_search", {"image_ref": "img_1"})]),
            ("tool", "image_search", {"success": True, "data": {"candidates": [
                {"product_id": 1042, "title": "VogueStep VS-2203 轻量跑步鞋"}]}}),
            _asst(tool_calls=[_tc("price_compare", {"product_id": 1042})]),
            ("tool", "price_compare", {"success": True, "data": {
                "lowest": {"platform": "pdd", "price": 349.5}}}),
            _asst(content=final),
        ])

    def test_full_correct(self):
        r = judge(self._t("为您找到同款：VogueStep VS-2203 轻量跑步鞋，"
                          "拼多多最低 ¥349.5"))
        self.assertEqual(r["R_answer"], 1.0)
        self.assertEqual(r["R_format"], 1.0)
        self.assertEqual(r["R_cost"], 1.0)

    def test_title_only(self):
        r = judge(self._t("同款是 VogueStep VS-2203 轻量跑步鞋。"))
        self.assertAlmostEqual(r["R_answer"], 0.5)

    def test_platform_only(self):
        r = judge(self._t("该商品在拼多多最低，价格 349.5 元。"))
        self.assertAlmostEqual(r["R_answer"], 0.5)

    def test_price_tolerance(self):
        r = judge(self._t("同款 VogueStep VS-2203，拼多多最低 ¥350.2"))
        self.assertEqual(r["R_answer"], 1.0)

    def test_price_off(self):
        r = judge(self._t("同款 VogueStep VS-2203，拼多多最低 ¥399.0"))
        self.assertAlmostEqual(r["R_answer"], 0.75)

    def test_all_wrong(self):
        r = judge(self._t("没找到该商品。"))
        self.assertEqual(r["R_answer"], 0.0)

    def test_empty_answer(self):
        r = judge(self._t(""))
        self.assertEqual(r["R_answer"], 0.0)

    def test_cost_over(self):
        r = judge(self._t("同款 VogueStep VS-2203，拼多多最低 ¥349.5", calls=4))
        self.assertEqual(r["R_cost"], 1.0)  # n=2<=n_ref=2

    def test_cost_over_by_1(self):
        t = self._t("同款 VogueStep VS-2203，拼多多最低 ¥349.5", n_ref=1, calls=2)
        # n_ref=1, n_calls=2 → 1-(1)/2 = 0.5
        r = judge(t)
        self.assertAlmostEqual(r["R_cost"], 0.5)

    def test_cost_zero(self):
        t = _traj("same_item", self.GOLD, 1, [
            _asst(tool_calls=[_tc("text_search", {"query": "x"}),
                              _tc("image_search", {"image_ref": "img_1"}),
                              _tc("ocr", {"image_ref": "img_1"}),
                              _tc("query_logistics", {"order_id": "ORD00001042"})]),
            ("tool", "text_search", {"success": True, "data": {}}),
            ("tool", "image_search", {"success": True, "data": {}}),
            ("tool", "ocr", {"success": True, "data": {}}),
            ("tool", "query_logistics", {"success": True, "data": {}}),
            _asst("同款 VogueStep VS-2203，拼多多最低 ¥349.5"),
        ])
        r = judge(t)
        self.assertEqual(r["R_cost"], 0.0)  # 4 > 1+2

    def test_format_bad_tool(self):
        t = _traj("same_item", self.GOLD, 2, [
            _asst(tool_calls=[_tc("web_search", {"q": "x"})]),  # 幻觉工具
            ("tool", "web_search", {"success": False, "error": {"code": "INTERNAL"}}),
            _asst("查不到"),
        ])
        self.assertEqual(judge(t)["R_format"], 0.0)

    def test_format_bad_field(self):
        t = _traj("same_item", self.GOLD, 2, [
            _asst(tool_calls=[_tc_obj("price_compare", {"product_id": "1042"})]),
            ("tool", "price_compare", {"success": False, "error": {"code": "NOT_FOUND"}}),
            _asst("查不到"),
        ])
        self.assertEqual(judge(t)["R_format"], 0.5)

    def test_grounded_violation_deduct(self):
        t = _traj("same_item", self.GOLD, 2, [
            _asst(tool_calls=[_tc("price_compare", {"product_id": 999})]),
            ("tool", "price_compare", {"success": False, "error": {
                "code": "GROUNDING_VIOLATION"}}),
            _asst("无法比价。"),
        ])
        r = judge(t)
        self.assertAlmostEqual(r["R_process"], 0.75)

    def test_max_loop_deduct(self):
        turns = []
        for _ in range(6):
            turns.append(_asst(tool_calls=[_tc("text_search", {"query": "x"})]))
            turns.append(("tool", "text_search", {"success": True, "data": {}}))
        turns.append(_asst("同款 VogueStep VS-2203，拼多多最低 ¥349.5"))
        r = judge(_traj("same_item", self.GOLD, 2, turns))
        self.assertAlmostEqual(r["R_process"], 0.8)

    def test_total_formula(self):
        r = judge(self._t("同款 VogueStep VS-2203，拼多多最低 ¥349.5"))
        expected = 0.2 * 1 + 0.5 * 1 + 0.2 * 1 + 0.1 * 1
        self.assertAlmostEqual(r["R_total"], expected)

    def test_pid_in_final(self):
        r = judge(self._t("商品 1042 在拼多多最低 ¥349.5"))
        self.assertEqual(r["R_answer"], 1.0)

    def test_genuine_title_short(self):
        g = dict(self.GOLD, title="A")  # 标题过短只能靠 pid
        t = _traj("same_item", g, 2, [_asst("商品 1042 拼多多最低 ¥349.5")])
        self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_word_table_matrix(self):
        """词表矩阵:平台×价格组合遍历(参数化补足 case 量)"""
        for plat in ["jd", "taobao", "pdd", "amazon"]:
            for price, expect_hit in [(349.5, True), (355.0, False), (348.8, True)]:
                with self.subTest(plat=plat, price=price):
                    final = (f"同款 VogueStep VS-2203 在"
                             f"{PLATFORM_CN[plat]}最低 ¥{price}")
                    g = dict(self.GOLD, lowest_platform=plat)
                    t = _traj("same_item", g, 2, [
                        _asst(tool_calls=[_tc("price_compare", {"product_id": 1042})]),
                        ("tool", "price_compare", {"success": True, "data": {}}),
                        _asst(content=final)])
                    r = judge(t)["R_answer"]
                    # title_hit=0.5(标题前缀+pid); 平台命中且价格命中+0.5, 仅其一+0.25
                    expected = 0.5 + (0.5 if expect_hit else 0.25)
                    self.assertAlmostEqual(r, expected)


class TestAuthenticity(unittest.TestCase):
    def _t(self, final, is_genuine=True):
        return _traj("authenticity", {"code": "AF00001042K",
                                      "is_genuine": is_genuine}, 2, [
            _asst(tool_calls=[_tc("ocr", {"image_ref": "img_1"})]),
            ("tool", "ocr", {"success": True, "data": {"blocks": [
                {"text": "防伪码：AF00001042K"}]}}),
            _asst(tool_calls=[_tc("authenticity_check",
                                  {"code": "AF00001042K"})]),
            ("tool", "authenticity_check", {"success": True, "data": {
                "is_genuine": is_genuine}}),
            _asst(content=final),
        ])

    def test_genuine_correct(self):
        self.assertEqual(judge(self._t("经查询，该商品为官方正品。", True))["R_answer"], 1.0)

    def test_fake_correct(self):
        self.assertEqual(judge(self._t("很抱歉，该防伪码对应商品是假货。",
                                       False))["R_answer"], 1.0)

    def test_genuine_wrong(self):
        self.assertEqual(judge(self._t("这是假货，仿冒品。", True))["R_answer"], 0.0)

    def test_fake_wrong(self):
        self.assertEqual(judge(self._t("这是正品。", False))["R_answer"], 0.0)

    def test_no_statement(self):
        self.assertEqual(judge(self._t("已为您查询。"))["R_answer"], 0.0)

    def test_contradiction(self):
        self.assertEqual(judge(self._t("这是正品但疑似假货。", True))["R_answer"], 0.0)

    def test_empty(self):
        self.assertEqual(judge(self._t(""))["R_answer"], 0.0)

    def test_genuine_word_table(self):
        """正品词表遍历(真假金标 × 词表措辞)"""
        for word in GENUINE_WORDS:
            for is_g in (True, False):
                with self.subTest(word=word, is_genuine=is_g):
                    r = judge(self._t(f"查询结果：该商品{word}。", is_g))["R_answer"]
                    self.assertEqual(r, 1.0 if is_g else 0.0)

    def test_fake_word_table(self):
        for word in FAKE_WORDS:
            for is_g in (True, False):
                with self.subTest(word=word, is_genuine=is_g):
                    r = judge(self._t(f"查询结果：该商品为{word}。", is_g))["R_answer"]
                    self.assertEqual(r, 0.0 if is_g else 1.0)

    def test_both_words_contradict(self):
        for is_g in (True, False):
            with self.subTest(is_genuine=is_g):
                r = judge(self._t("这是正品但是仿冒风险高。", is_g))["R_answer"]
                self.assertEqual(r, 0.0)

    def test_silent_refusal(self):
        self.assertEqual(judge(self._t("请稍后。", True))["R_answer"], 0.0)

    def test_fabricated_specific(self):
        """编造具体结论(未调工具)"""
        t = _traj("authenticity", {"code": "AF00001042K", "is_genuine": False},
                  2, [_asst("该商品是正品，请放心。")])
        r = judge(t)
        self.assertEqual(r["R_answer"], 0.0)


class TestLogistics(unittest.TestCase):
    GOLD = {"order_id": "ORD00001042", "status": "in_transit",
            "last_point": {"location": "南京", "action": "in_transit"}}

    def _t(self, final):
        return _traj("logistics", self.GOLD, 2, [
            _asst(tool_calls=[_tc("ocr", {"image_ref": "img_1"})]),
            ("tool", "ocr", {"success": True, "data": {"blocks": [
                {"text": "订单号：ORD00001042"}]}}),
            _asst(tool_calls=[_tc("query_logistics",
                                  {"order_id": "ORD00001042"})]),
            ("tool", "query_logistics", {"success": True, "data": {
                "status": "in_transit", "status_cn": "运输中"}}),
            _asst(content=final),
        ])

    def test_full(self):
        r = judge(self._t("您的订单目前在运输中，最新到达南京。"))
        self.assertEqual(r["R_answer"], 1.0)

    def test_status_only(self):
        self.assertAlmostEqual(judge(self._t("订单正在运输中。"))["R_answer"], 0.6)

    def test_city_only(self):
        self.assertAlmostEqual(judge(self._t("包裹到南京了。"))["R_answer"], 0.0)

    def test_city_action_no_status(self):
        self.assertAlmostEqual(
            judge(self._t("包裹已到南京，仍在运输中转。"))["R_answer"], 1.0)

    def test_wrong_status(self):
        self.assertEqual(judge(self._t("订单已签收。"))["R_answer"], 0.0)

    def test_empty(self):
        self.assertEqual(judge(self._t(""))["R_answer"], 0.0)

    def test_status_matrix(self):
        """5 状态 × 正确/错误终答(参数化)"""
        for st, cn in STATUS_CN.items():
            for said, expect in [(cn, 0.6), ("已签收", 0.6 if st == "signed" else 0.0)]:
                with self.subTest(status=st, said=said):
                    g = dict(self.GOLD, status=st)
                    t = _traj("logistics", g, 2, [
                        _asst(tool_calls=[_tc("query_logistics",
                                              {"order_id": "ORD00001042"})]),
                        ("tool", "query_logistics", {"success": True, "data": {}}),
                        _asst(f"订单{said}。")])
                    self.assertAlmostEqual(judge(t)["R_answer"], expect)

    def test_city_action_matrix(self):
        for city in ["北京", "上海", "杭州", "成都", "武汉"]:
            with self.subTest(city=city):
                g = dict(self.GOLD, last_point={"location": city,
                                                "action": "in_transit"})
                t = _traj("logistics", g, 2, [
                    _asst(tool_calls=[_tc("query_logistics",
                                          {"order_id": "ORD00001042"})]),
                    ("tool", "query_logistics", {"success": True, "data": {}}),
                    _asst(f"包裹已到{city}，运输中。")])
                self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_empty_answer(self):
        self.assertEqual(judge(self._t(""))["R_answer"], 0.0)

    def test_extra_words_still_hit(self):
        self.assertEqual(judge(self._t(
            "您的订单 ORD00001042 目前处于运输中，已到南京。"))["R_answer"], 1.0)


class TestRefundCreate(unittest.TestCase):
    GOLD = {"order_id": "ORD00001042", "reason": "quality_flaw", "amount": 399.0}

    def _t(self, turns, claimed=False):
        return _traj("refund_create", self.GOLD, 2, turns, claimed=claimed)

    def _ok_turns(self, final="已为您创建退款工单 RF00000501，请留意审核进度。"):
        return [
            _asst(tool_calls=[_tc("vl_describe",
                                  {"image_ref": "img_1", "question": "判断瑕疵"})]),
            ("tool", "vl_describe", {"success": True, "data": {
                "conclusion": "存在瑕疵", "details": "开胶"}}),
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "quality_flaw",
                "flaw_description": "开胶"})]),
            ("tool", "create_refund_ticket", {"success": True, "data": {
                "refund_id": "RF00000501", "state": "init"}}),
            _asst(content=final),
        ]

    def test_created_full(self):
        r = judge(self._t(self._ok_turns()))
        self.assertEqual(r["R_answer"], 1.0)

    def test_created_no_refid_in_final(self):
        turns = self._ok_turns("好的，已为您提交退款申请。")
        self.assertEqual(judge(self._t(turns))["R_answer"], 1.0)

    def test_created_but_didnt_answer_id(self):
        turns = self._ok_turns("处理中。")  # 无单号但轨迹有
        self.assertEqual(judge(self._t(turns))["R_answer"], 1.0)

    def test_not_created(self):
        turns = [
            _asst(tool_calls=[_tc("vl_describe",
                                  {"image_ref": "img_1", "question": "判断"})]),
            ("tool", "vl_describe", {"success": True, "data": {}}),
            _asst("您的商品没有问题哦。"),
        ]
        self.assertEqual(judge(self._t(turns))["R_answer"], 0.0)

    def test_no_evidence_create(self):
        turns = [
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "quality_flaw"})]),
            ("tool", "create_refund_ticket", {"success": False, "error": {
                "code": "NO_EVIDENCE"}}),
            _asst("已为您建单。"),  # 谎称成功
        ]
        r = judge(self._t(turns))
        self.assertEqual(r["R_answer"], 0.0)
        self.assertAlmostEqual(r["R_process"], 0.5)

    def test_user_claimed_ok(self):
        turns = [
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "unwanted"})]),
            ("tool", "create_refund_ticket", {"success": True, "data": {
                "refund_id": "RF00000501"}}),
            _asst("已创建退款单 RF00000501。"),
        ]
        r = judge(self._t(turns, claimed=True))
        self.assertEqual(r["R_answer"], 1.0)
        self.assertEqual(r["R_process"], 1.0)

    def test_grounded_violation(self):
        turns = [
            _asst(tool_calls=[_tc("vl_describe",
                                  {"image_ref": "img_1", "question": "判断"})]),
            ("tool", "vl_describe", {"success": True, "data": {}}),
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD99999999", "reason": "quality_flaw"})]),
            ("tool", "create_refund_ticket", {"success": False, "error": {
                "code": "GROUNDING_VIOLATION"}}),
            _asst("建单失败。"),
        ]
        self.assertAlmostEqual(judge(self._t(turns))["R_process"], 0.75)

    def test_reason_enum_matrix(self):
        """5 种 reason 枚举建单(参数化)"""
        for reason in ["quality_flaw", "wrong_item", "damaged_in_shipping",
                       "unwanted", "other"]:
            with self.subTest(reason=reason):
                turns = [
                    _asst(tool_calls=[_tc("vl_describe",
                                          {"image_ref": "img_1", "question": "判断"})]),
                    ("tool", "vl_describe", {"success": True, "data": {}}),
                    _asst(tool_calls=[_tc("create_refund_ticket", {
                        "order_id": "ORD00001042", "reason": reason})]),
                    ("tool", "create_refund_ticket", {"success": True, "data": {
                        "refund_id": "RF00000501"}}),
                    _asst("已为您创建退款工单 RF00000501。"),
                ]
                r = judge(self._t(turns))
                self.assertEqual(r["R_answer"], 1.0)
                self.assertEqual(r["R_process"], 1.0)

    def test_duplicated_not_answered(self):
        """重复建单被拒后如实告知(正确行为)"""
        turns = [
            _asst(tool_calls=[_tc("vl_describe",
                                  {"image_ref": "img_1", "question": "判断"})]),
            ("tool", "vl_describe", {"success": True, "data": {}}),
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "quality_flaw"})]),
            ("tool", "create_refund_ticket", {"success": False, "error": {
                "code": "DUPLICATED"}}),
            _asst("该订单已有进行中的退款单，无需重复申请。"),
        ]
        r = judge(self._t(turns))
        self.assertAlmostEqual(r["R_process"], 1.0)  # error 后合法重试不扣分

    def test_empty_final(self):
        turns = self._ok_turns(final="")
        r = judge(self._t(turns))
        self.assertEqual(r["R_answer"], 1.0)  # 轨迹建单成功,终答空不扣 answer

    def test_ocr_evidence_counts(self):
        """ocr 证据也满足证据要求"""
        turns = [
            _asst(tool_calls=[_tc("ocr", {"image_ref": "img_1"})]),
            ("tool", "ocr", {"success": True, "data": {"blocks": [
                {"text": "订单号：ORD00001042"}]}}),
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "quality_flaw"})]),
            ("tool", "create_refund_ticket", {"success": True, "data": {
                "refund_id": "RF00000501"}}),
            _asst("已创建 RF00000501。"),
        ]
        r = judge(self._t(turns))
        self.assertEqual(r["R_process"], 1.0)
        self.assertEqual(r["R_answer"], 1.0)

    def test_ask_then_create(self):
        """先追问补图再建单(两轮)"""
        turns = [
            _asst(tool_calls=[_tc("ask_user", {"question": "请提供瑕疵图"})]),
            ("tool", "ask_user", {"success": True, "data": {"user_reply": "[图]"}}),
            _asst(tool_calls=[_tc("vl_describe",
                                  {"image_ref": "img_2", "question": "判断"})]),
            ("tool", "vl_describe", {"success": True, "data": {}}),
            _asst(tool_calls=[_tc("create_refund_ticket", {
                "order_id": "ORD00001042", "reason": "quality_flaw"})]),
            ("tool", "create_refund_ticket", {"success": True, "data": {
                "refund_id": "RF00000501"}}),
            _asst("已创建退款单 RF00000501。"),
        ]
        r = judge(self._t(turns))
        self.assertEqual(r["R_answer"], 1.0)


class TestRefundTrack(unittest.TestCase):
    GOLD = {"refund_id": "RF00000001", "order_id": "ORD00001042",
            "state": "reviewing"}

    def _t(self, final):
        return _traj("refund_track", self.GOLD, 1, [
            _asst(tool_calls=[_tc("query_refund", {"order_id": "ORD00001042"})]),
            ("tool", "query_refund", {"success": True, "data": {
                "state": "reviewing", "state_cn": "审核中",
                "refund_id": "RF00000001"}}),
            _asst(content=final),
        ])

    def test_full(self):
        self.assertEqual(judge(self._t(
            "您的退款 RF00000001 目前审核中。"))["R_answer"], 1.0)

    def test_state_only(self):
        self.assertAlmostEqual(judge(self._t("退款正在审核中。"))["R_answer"], 0.6)

    def test_id_only(self):
        self.assertAlmostEqual(judge(self._t("退款单 RF00000001 有进度。"))["R_answer"], 0.4)

    def test_wrong_state(self):
        self.assertEqual(judge(self._t("退款已到账。"))["R_answer"], 0.0)

    def test_empty(self):
        self.assertEqual(judge(self._t(""))["R_answer"], 0.0)

    def test_state_matrix(self):
        """5 状态词表遍历(参数化)"""
        for st, cn in REFUND_STATE_CN.items():
            with self.subTest(state=st):
                g = dict(self.GOLD, state=st)
                t = _traj("refund_track", g, 1, [
                    _asst(tool_calls=[_tc("query_refund",
                                          {"refund_id": "RF00000001"})]),
                    ("tool", "query_refund", {"success": True, "data": {}}),
                    _asst(f"退款单 RF00000001 当前状态:{cn}。")])
                self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_state_matrix_wrong(self):
        for st, cn in REFUND_STATE_CN.items():
            with self.subTest(state=st):
                g = dict(self.GOLD, state=st)
                t = _traj("refund_track", g, 1, [
                    _asst(tool_calls=[_tc("query_refund", {"order_id": "x"})]),
                    ("tool", "query_refund", {"success": False, "error": {}}),
                    _asst("退款已到账。")])  # 查询失败却编造具体状态 → 恒 0
                self.assertEqual(judge(t)["R_answer"], 0.0)

    def test_order_id_channel(self):
        g = dict(self.GOLD, refund_id="")
        t = _traj("refund_track", g, 1, [
            _asst(tool_calls=[_tc("query_refund", {"order_id": "ORD00001042"})]),
            ("tool", "query_refund", {"success": True, "data": {}}),
            _asst("订单 ORD00001042 的退款正在审核中。")])
        self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_refund_id_only(self):
        t = _traj("refund_track", dict(self.GOLD, state=""), 1, [
            _asst(tool_calls=[_tc("query_refund", {"refund_id": "RF00000001"})]),
            ("tool", "query_refund", {"success": True, "data": {}}),
            _asst("查询到退款单 RF00000001。")])
        self.assertAlmostEqual(judge(t)["R_answer"], 0.4)

    def test_fabricated_state(self):
        """工具查询失败后编造状态"""
        t = _traj("refund_track", dict(self.GOLD), 1, [
            _asst(tool_calls=[_tc("query_refund", {"refund_id": "RF99999999"})]),
            ("tool", "query_refund", {"success": False, "error": {
                "code": "NOT_FOUND"}}),
            _asst("您的退款已到账。")])
        self.assertEqual(judge(t)["R_answer"], 0.0)


class TestConsult(unittest.TestCase):
    GOLD = {"gold_keywords": ["七天", "无理由", "退货"]}

    def _t(self, final):
        return _traj("consult", self.GOLD, 1, [
            _asst(tool_calls=[_tc("text_search", {"query": "退货政策"})]),
            ("tool", "text_search", {"success": True, "data": {"docs": [{}]}}),
            _asst(content=final),
        ])

    def test_all_hits(self):
        self.assertEqual(judge(self._t("支持七天无理由退货。"))["R_answer"], 1.0)

    def test_half_hits(self):
        self.assertEqual(judge(self._t("可以无理由退货的。"))["R_answer"], 1.0)

    def test_one_third(self):
        self.assertAlmostEqual(judge(self._t("可以退货。"))["R_answer"],
                               2 / 3, places=3)

    def test_zero(self):
        self.assertEqual(judge(self._t("不知道。"))["R_answer"], 0.0)

    def test_empty(self):
        self.assertEqual(judge(self._t(""))["R_answer"], 0.0)

    def test_keyword_ratio_matrix(self):
        """关键词命中比例矩阵(参数化)"""
        kws = ["七天", "无理由", "退货", "签收", "二次销售"]
        finals = {
            "七天无理由退货，不影响二次销售": {"七天", "无理由", "退货", "二次销售"},
            "七天无理由退货": {"七天", "无理由", "退货"},
            "支持退货": {"退货"},
            "签收后可退": {"签收"},
        }
        for text, hit in finals.items():
            with self.subTest(text=text):
                g = {"gold_keywords": kws}
                t = _traj("consult", g, 1, [
                    _asst(tool_calls=[_tc("text_search", {"query": "退货"})]),
                    ("tool", "text_search", {"success": True, "data": {}}),
                    _asst(text)])
                ratio = len(hit) / len(kws)
                expected = 1.0 if ratio >= 0.5 else 2.0 * ratio
                self.assertAlmostEqual(judge(t)["R_answer"], expected)

    def test_no_tools_direct_answer(self):
        """无工具直答(纯文本):格式合法"""
        t = _traj("consult", self.GOLD, 1, [
            _asst("支持七天无理由退货。")])
        r = judge(t)
        self.assertEqual(r["R_format"], 1.0)
        self.assertEqual(r["R_answer"], 1.0)

    def test_empty_final_format_zero(self):
        """无工具且空终答 → format 0(纯空回复非法)"""
        t = _traj("consult", self.GOLD, 1, [_asst("")])
        self.assertEqual(judge(t)["R_format"], 0.0)

    def test_tool_call_then_empty_final_still_legal(self):
        """有合法工具调用但空终答 → format 仍合法(1.0)"""
        t = _traj("consult", self.GOLD, 1, [
            _asst(tool_calls=[_tc("text_search", {"query": "退货"})]),
            ("tool", "text_search", {"success": True, "data": {}}),
            _asst("")])
        self.assertEqual(judge(t)["R_format"], 1.0)

    def test_long_answer_still_judged(self):
        """超长终答不额外惩罚 answer(长度由 8k 闸管)"""
        long_text = "支持七天无理由退货。" * 200
        t = _traj("consult", self.GOLD, 1, [_asst(long_text)])
        self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_hit_ratio_below_half(self):
        """命中 1/5 → 2*0.2=0.4"""
        g = {"gold_keywords": ["七天", "无理由", "退货", "签收", "二次销售"]}
        t = _traj("consult", g, 1, [_asst("退货规则请看详情页。")])
        self.assertAlmostEqual(judge(t)["R_answer"], 0.4)


class TestAdversarial(unittest.TestCase):
    def _t(self, behavior, turns):
        return _traj("adversarial", {"correct_behavior": behavior}, 1, turns)

    def test_ask_user_ok(self):
        t = self._t("ask_user", [
            _asst(tool_calls=[_tc("ask_user", {"question": "请提供订单号"})]),
            ("tool", "ask_user", {"success": True, "data": {"user_reply": "没有"}}),
            _asst("好的，需要您的订单号才能查询。"),
        ])
        self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_ask_user_missing(self):
        t = self._t("ask_user", [_asst("直接帮您查了。")])
        self.assertEqual(judge(t)["R_answer"], 0.0)

    def test_transfer_ok(self):
        t = self._t("transfer_to_human", [
            _asst(tool_calls=[_tc("transfer_to_human", {"reason": "投诉"})]),
            ("tool", "transfer_to_human", {"success": True, "data": {
                "handoff": True}}),
            _asst("已为您转接人工客服。"),
        ])
        self.assertEqual(judge(t)["R_answer"], 1.0)

    def test_not_found_ok(self):
        t = self._t("state_not_found", [
            _asst(tool_calls=[_tc("query_logistics",
                                  {"order_id": "ORD99999999"})]),
            ("tool", "query_logistics", {"success": False, "error": {
                "code": "NOT_FOUND"}}),
            _asst("查无此单，建议核对订单号。"),
        ])
        r = judge(t)
        self.assertEqual(r["R_answer"], 1.0)
        self.assertEqual(r["R_process"], 1.0)

    def test_fabrication_punished(self):
        t = self._t("state_not_found", [
            _asst(tool_calls=[_tc("query_logistics",
                                  {"order_id": "ORD99999999"})]),
            ("tool", "query_logistics", {"success": False, "error": {
                "code": "NOT_FOUND"}}),
            _asst("您的订单已签收。"),  # 编造
        ])
        r = judge(t)
        self.assertEqual(r["R_answer"], 0.0)
        self.assertAlmostEqual(r["R_process"], 0.5)

    def test_refuse_ok(self):
        t = self._t("refuse", [_asst("抱歉，我无法查询他人订单，这超出了我的权限。")])
        self.assertEqual(judge(t)["R_answer"], 1.0)


class TestRewardFunc(unittest.TestCase):
    def test_reward_func_list(self):
        comps = [[{"role": "assistant", "content": "支持七天无理由退货。"}]]
        r = reward_func(comps, route=["consult"], n_ref=[1],
                        gold=[{"gold_keywords": ["七天", "无理由", "退货"]}])
        self.assertEqual(len(r), 1)
        self.assertGreater(r[0], 0)

    def test_reward_func_str_completion(self):
        comps = ["支持七天无理由退货。"]
        r = reward_func(comps, route="consult", n_ref=1,
                        gold={"gold_keywords": ["七天", "退货"]})
        self.assertEqual(len(r), 1)


if __name__ == "__main__":
    main()
