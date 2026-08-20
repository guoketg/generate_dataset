"""train_executor.py —— 训练态工具执行器(PRD 06 5.10 双形态之训练态)。

读 17 全部数据资产 JSONL 建内存索引(总量 < 50MB),实现 11 工具真实执行语义;
参数溯源校验(PRD 06 5.5)、Observation 统一信封(5.4,≤2KB 截断)、
写入类风控(5.8)。Teacher 轨迹生成与 GRPO Rollout 共用本执行器。

注意:感知类工具(ocr/vl_describe)的"真实执行"= 合成图生成时落盘的
元数据 ground truth(meta.jsonl),非凭空模拟;结构化查询类工具全部真实
查内存索引。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # cloud_training/
DATA = _ROOT / "data"

# ---------------- 内置知识库(text_search 依赖;17 资产未含 FAQ/政策文档,
# ---------------- 由本模块内置确定性文档,见 decisions.log)----------------
FAQ_POLICIES = [
    {"source_type": "policy", "title": "七天无理由退货",
     "content": "自签收之日起7天内，商品未使用且不影响二次销售的，可申请无理由退货。定制类、生鲜类商品除外。",
     "keywords": ["七天", "无理由", "退货", "签收"]},
    {"source_type": "policy", "title": "十五天质量问题换货",
     "content": "自签收之日起15天内，商品出现非人为质量问题，可选择换货或维修，运费由商家承担。",
     "keywords": ["质量问题", "换货", "十五天", "运费"]},
    {"source_type": "policy", "title": "保修政策",
     "content": "数码家电类商品享受一年整机保修，主要部件保修两年。人为损坏、进水、私自拆修不在保修范围。",
     "keywords": ["保修", "一年", "维修", "拆修"]},
    {"source_type": "policy", "title": "假一赔十",
     "content": "经防伪码验证确认为假货的，平台执行假一赔十赔付政策，并先行垫付退款。",
     "keywords": ["假货", "赔十", "防伪", "赔付"]},
    {"source_type": "policy", "title": "运费险说明",
     "content": "退货时可使用运费险抵扣首重运费，每单最高赔付12元，超重部分需用户自理。",
     "keywords": ["运费险", "退货", "12元", "首重"]},
    {"source_type": "policy", "title": "退款到账时间",
     "content": "退款审核通过后，原支付渠道1-3个工作日到账，信用卡渠道可能延迟至5个工作日。",
     "keywords": ["退款", "到账", "工作日", "信用卡"]},
    {"source_type": "policy", "title": "发货时效",
     "content": "现货商品下单后48小时内发货，预售商品按商品页标注时间发货，大促期间可能延迟24小时。",
     "keywords": ["发货", "48小时", "预售", "大促"]},
    {"source_type": "policy", "title": "价保政策",
     "content": "下单后7天内商品降价，可申请差价补偿。百亿补贴、秒杀、直播专属价不参与价保。",
     "keywords": ["价保", "降价", "差价", "补贴"]},
    {"source_type": "faq", "title": "如何查询物流",
     "content": "在订单详情页可查看实时物流轨迹，也可联系客服提供订单号查询。",
     "keywords": ["物流", "订单", "轨迹", "查询"]},
    {"source_type": "faq", "title": "如何查询退款进度",
     "content": "提供订单号或退款单号，客服可查询退款工单当前状态(init/reviewing/approved/refunded/rejected)。",
     "keywords": ["退款", "进度", "工单", "状态"]},
    {"source_type": "faq", "title": "防伪码在哪里",
     "content": "防伪码通常印在商品包装盒侧面或标签上，刮开涂层即可查看，也可拍照发给客服识别。",
     "keywords": ["防伪码", "包装", "标签", "涂层"]},
    {"source_type": "faq", "title": "发票开具",
     "content": "支持电子发票，下单时在备注栏填写开票信息，或收货后联系客服补开。",
     "keywords": ["发票", "电子", "开票", "备注"]},
    {"source_type": "faq", "title": "修改收货地址",
     "content": "订单未发货前可修改收货地址，发货后无法修改，可联系快递公司改派。",
     "keywords": ["地址", "修改", "发货", "快递"]},
    {"source_type": "faq", "title": "尺码建议",
     "content": "服饰类商品详情页有尺码对照表，建议按身高体重选择，两个尺码之间建议选大一码。",
     "keywords": ["尺码", "身高", "体重", "对照表"]},
    {"source_type": "faq", "title": "同款比价说明",
     "content": "提供商品图片或商品信息，客服可通过比价工具查询京东、淘宝、拼多多、亚马逊四平台价格。",
     "keywords": ["比价", "平台", "价格", "同款"]},
    {"source_type": "faq", "title": "拒收说明",
     "content": "商品未签收前可拒收，拒收后商品退回，运费问题视拒收原因而定。",
     "keywords": ["拒收", "签收", "退回", "运费"]},
]

PLATFORM_CN = {"jd": "京东", "taobao": "淘宝", "pdd": "拼多多", "amazon": "亚马逊"}
STATUS_CN = {"shipped": "已发货", "in_transit": "运输中", "delivering": "派送中",
             "signed": "已签收", "rejected": "已拒收"}
REFUND_STATE_CN = {"init": "已发起待审核", "reviewing": "审核中",
                   "approved": "审核通过待到账", "refunded": "已退款",
                   "rejected": "已拒绝"}
MAX_OBS_BYTES = 2048
ACTIVE_REFUND_STATES = {"init", "reviewing", "approved"}
ORDER_ID_RE = re.compile(r"ORD\d{8}")
REFUND_ID_RE = re.compile(r"RF\d{8}")
AF_CODE_RE = re.compile(r"AF\d{8}[A-Z]")


def envelope(ok: bool, data=None, code: str = "", message: str = "") -> dict:
    """Observation 统一信封(PRD 06 5.4)"""
    if ok:
        return {"success": True, "data": data or {}}
    return {"success": False, "error": {"code": code, "message": message}}


def truncate_obs(obs: dict) -> str:
    """序列化 + ≤2KB 截断标记"""
    s = json.dumps(obs, ensure_ascii=False)
    if len(s.encode("utf-8")) <= MAX_OBS_BYTES:
        return s
    while len(json.dumps(obs, ensure_ascii=False).encode("utf-8")) > MAX_OBS_BYTES - 32:
        obs = json.loads(json.dumps(obs, ensure_ascii=False))
        data = obs.get("data")
        if isinstance(data, dict):
            for k in list(data.keys()):
                v = data.pop(k)
                if isinstance(v, list) and len(v) > 1:
                    v = v[: max(1, len(v) // 2)]
                    data[k] = v
                    break
                if isinstance(v, str) and len(v) > 40:
                    data[k] = v[:40]
                    break
            else:
                break
        else:
            break
    obs["truncated"] = True
    return json.dumps(obs, ensure_ascii=False)


class TrainExecutor:
    """训练态执行器:每条轨迹新建一个实例(session 级状态隔离)。"""

    def __init__(self, data_dir: str = "data", rng: random.Random | None = None):
        self.rng = rng or random.Random(42)
        self.products: dict[int, dict] = {}
        self.products_by_cat: dict[str, list] = {}
        self.prices: dict[int, dict] = {}
        self.anti_fake: dict[str, dict] = {}
        self.logistics: dict[str, dict] = {}
        self.refunds: dict[str, dict] = {}
        self.refunds_by_order: dict[str, list] = {}
        self._load(Path(data_dir))
        # ---- 会话状态 ----
        self.images: dict[str, dict] = {}      # img_ref -> meta
        self.grounded: set = set()             # 合法参数值(溯源)
        self.has_evidence = False              # vl_describe/ocr 证据
        self.user_claimed_refund = False       # 用户明确陈述退款
        self.sim_user_replies: list[str] = []  # ask_user 模拟回复队列
        self.created_refunds: dict[str, str] = {}  # order_id -> refund_id
        self.next_refund_id = 500
        self.finished = False                  # transfer_to_human 终止标记
        self.n_calls = 0

    # ---------------- 数据索引 ----------------
    def _load(self, data_dir: Path):
        for line in open(data_dir / "products.jsonl", encoding="utf-8"):
            p = json.loads(line)
            self.products[p["product_id"]] = p
            self.products_by_cat.setdefault(p["category"], []).append(p)
        for line in open(data_dir / "prices.jsonl", encoding="utf-8"):
            r = json.loads(line)
            self.prices.setdefault(r["product_id"], {})[r["platform"]] = r["price"]
        for line in open(data_dir / "anti_fake.jsonl", encoding="utf-8"):
            a = json.loads(line)
            self.anti_fake[a["code"]] = a
        for line in open(data_dir / "logistics.jsonl", encoding="utf-8"):
            t = json.loads(line)
            self.logistics[t["order_id"]] = t
        for line in open(data_dir / "refunds.jsonl", encoding="utf-8"):
            r = json.loads(line)
            self.refunds[r["refund_id"]] = r
            self.refunds_by_order.setdefault(r["order_id"], []).append(r)

    # ---------------- 会话注册 ----------------
    def register_image(self, img_ref: str, meta: dict):
        """注册会话图片(img_N -> 元数据)。meta 至少含 type。"""
        self.images[img_ref] = meta

    def observe_user_text(self, text: str):
        """用户文本入溯源池:提取单号/防伪码/数字ID"""
        for m in ORDER_ID_RE.finditer(text):
            self.grounded.add(m.group(0))
        for m in REFUND_ID_RE.finditer(text):
            self.grounded.add(m.group(0))
        for m in AF_CODE_RE.finditer(text):
            self.grounded.add(m.group(0))
        for m in re.finditer(r"\b(?:商品\s*[IDid]{0,2}\s*[:：]?\s*)(\d{1,5})\b", text):
            pid = int(m.group(1))
            if pid in self.products:
                self.grounded.add(pid)

    def _ground_from_obs(self, obj):
        """把 Observation 中出现的可溯源值加入合法池"""
        s = json.dumps(obj, ensure_ascii=False)
        for m in ORDER_ID_RE.finditer(s):
            self.grounded.add(m.group(0))
        for m in REFUND_ID_RE.finditer(s):
            self.grounded.add(m.group(0))
        for m in AF_CODE_RE.finditer(s):
            self.grounded.add(m.group(0))
        for m in re.finditer(r'"product_id":\s*(\d+)', s):
            self.grounded.add(int(m.group(1)))

    # ---------------- 11 工具执行 ----------------
    def execute(self, name: str, args: dict) -> str:
        """统一入口:执行工具,返回 Observation JSON 字符串(role=tool content)。"""
        self.n_calls += 1
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            obs = envelope(False, code="INTERNAL",
                           message=f"未知工具: {name}")
        else:
            try:
                obs = handler(args or {})
            except Exception as e:  # noqa: BLE001 —— 执行异常兜底为 INTERNAL
                obs = envelope(False, code="INTERNAL", message=str(e)[:200])
        self._ground_from_obs(obs)
        return truncate_obs(obs)

    def _tool_ocr(self, args: dict) -> dict:
        img_ref = args.get("image_ref")
        if not img_ref:
            return envelope(False, code="MISSING_PARAM", message="缺少 image_ref")
        meta = self.images.get(img_ref)
        if meta is None:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message=f"会话中不存在图片 {img_ref}")
        focus = args.get("focus", "all")
        blocks = []
        if meta.get("type") == "order_screenshot":
            blocks = [
                {"text": f"订单号：{meta['order_id']}", "confidence": 0.98},
                {"text": f"金额：￥{meta['price']:.2f}", "confidence": 0.96},
                {"text": f"状态：{meta['status_cn']}", "confidence": 0.94},
            ]
        elif meta.get("type") == "anti_fake":
            blocks = [
                {"text": f"防伪码：{meta['code']}", "confidence": 0.97},
                {"text": f"商品 ID：{meta['product_id']}", "confidence": 0.95},
            ]
        elif meta.get("type") == "product":
            if meta.get("product_title"):
                blocks = [{"text": f"商品名称：{meta['product_title']}",
                           "confidence": 0.9}]
        if not blocks:
            return envelope(False, code="NO_TEXT", message="图片中未识别到文字")
        if focus and focus != "all":
            key = {"order_id": "订单号", "authenticity_code": "防伪码",
                   "amount": "金额"}.get(focus)
            if key:
                blocks = [b for b in blocks if b["text"].startswith(key)] or blocks
        self.has_evidence = True  # OCR 结果可视作证据
        return envelope(True, data={"blocks": blocks})

    def _tool_vl_describe(self, args: dict) -> dict:
        img_ref = args.get("image_ref")
        question = args.get("question", "")
        if not img_ref or not question:
            return envelope(False, code="MISSING_PARAM",
                            message="缺少 image_ref 或 question")
        meta = self.images.get(img_ref)
        if meta is None:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message=f"会话中不存在图片 {img_ref}")
        t = meta.get("type")
        if t == "defect":
            self.has_evidence = True
            return envelope(True, data={
                "conclusion": "存在瑕疵",
                "details": meta.get("defect_desc", "商品存在可见瑕疵"),
                "tags": ["defect", meta.get("defect_type", "瑕疵")],
            })
        if t == "product":
            self.has_evidence = True
            pid = meta.get("product_id")
            p = self.products.get(pid, {})
            return envelope(True, data={
                "conclusion": "商品图",
                "details": f"类目 {p.get('category', meta.get('category', ''))},"
                           f"品牌 {p.get('brand', '')},标题 {p.get('title', '')}",
                "tags": ["product", p.get("category", ""), p.get("brand", "")],
            })
        if t == "order_screenshot":
            return envelope(False, code="NOT_PRODUCT",
                            message="图片为订单截图，不是商品图")
        if t == "anti_fake":
            return envelope(True, data={
                "conclusion": "防伪码标签",
                "details": f"图中包含防伪码 {meta.get('code')}",
                "tags": ["anti_fake", "label"],
            })
        if t in ("blurry", "nature"):
            return envelope(False, code="NOT_PRODUCT" if t == "nature" else "BLURRY",
                            message="图片模糊不可辨识" if t == "blurry"
                            else "图片为风景/非商品图")
        return envelope(False, code="LOW_CONFIDENCE", message="无法判定图片内容")

    def _tool_image_search(self, args: dict) -> dict:
        img_ref = args.get("image_ref")
        if not img_ref:
            return envelope(False, code="MISSING_PARAM", message="缺少 image_ref")
        meta = self.images.get(img_ref)
        if meta is None:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message=f"会话中不存在图片 {img_ref}")
        top_k = min(max(int(args.get("top_k", 5)), 1), 10)
        category = args.get("category")
        t = meta.get("type")
        if t in ("order_screenshot", "anti_fake", "nature", "blurry"):
            return envelope(False, code="NOT_PRODUCT",
                            message="图片不是商品图，无法搜索同款")
        # 命中商品:图注册的 product_id(或缺陷图随机同类目)
        pid = meta.get("product_id")
        if pid is None:
            cat = meta.get("category") or self.rng.choice(list(self.products_by_cat))
            pool = self.products_by_cat[cat]
            pid = self.rng.choice(pool)["product_id"]
        hit = self.products[pid]
        cat = hit["category"]
        if category and category != cat:
            pool = self.products_by_cat.get(category, [])
            if not pool:
                return envelope(False, code="NO_MATCH", message="该类目无候选")
            hit = self.rng.choice(pool)
            pid, cat = hit["product_id"], hit["category"]
        # 同款 top1 + 同类目扰动候选
        cands = [{
            "product_id": pid, "title": hit["title"], "category": cat,
            "price": hit["price"],
            "similarity": round(self.rng.uniform(0.90, 0.97), 3),
        }]
        pool = [p for p in self.products_by_cat[cat] if p["product_id"] != pid]
        self.rng.shuffle(pool)
        for p in pool[: top_k - 1]:
            cands.append({
                "product_id": p["product_id"], "title": p["title"],
                "category": cat, "price": p["price"],
                "similarity": round(self.rng.uniform(0.70, 0.88), 3),
            })
        return envelope(True, data={"candidates": cands})

    def _tool_text_search(self, args: dict) -> dict:
        query = args.get("query")
        if not query:
            return envelope(False, code="MISSING_PARAM", message="缺少 query")
        scope = args.get("scope", "all")
        top_k = min(max(int(args.get("top_k", 5)), 1), 10)
        docs = []
        if scope in ("faq", "policy", "all"):
            for doc in FAQ_POLICIES:
                if doc["source_type"] == "policy" and scope == "faq":
                    continue
                if doc["source_type"] == "faq" and scope == "policy":
                    continue
                hits = sum(1 for k in doc["keywords"] if k in query)
                if hits or scope in ("faq", "policy", "all") and any(
                        k in query for k in doc["keywords"]):
                    docs.append({**doc, "score": round(0.6 + 0.1 * hits, 2)})
        if scope in ("product", "all"):
            for p in (self.products_by_cat.get("", []) or []):
                pass  # product 检索走下方标题匹配
            for p in list(self.products.values()):
                if any(w in p["title"] for w in query.split()):
                    docs.append({
                        "source_type": "product", "title": p["title"],
                        "content": f"product_id={p['product_id']}, 价格 {p['price']},"
                                   f" 类目 {p['category']}, 品牌 {p['brand']}",
                        "score": 0.8, "product_id": p["product_id"],
                    })
                if len(docs) >= 50:
                    break
        docs.sort(key=lambda d: -d.get("score", 0))
        if not docs:
            return envelope(False, code="NO_MATCH",
                            message="知识库无命中，建议换个说法")
        return envelope(True, data={"docs": docs[:top_k]})

    def _tool_price_compare(self, args: dict) -> dict:
        pid = args.get("product_id")
        if pid is None:
            return envelope(False, code="MISSING_PARAM", message="缺少 product_id")
        if pid not in self.products:
            return envelope(False, code="NOT_FOUND",
                            message=f"商品 {pid} 不存在，请核对 product_id")
        if pid not in self.grounded:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message="product_id 必须来自检索工具返回的候选，禁止编造")
        platforms = args.get("platforms") or ["jd", "taobao", "pdd", "amazon"]
        prices = [{"platform": p, "price": self.prices[pid][p]}
                  for p in platforms if p in self.prices.get(pid, {})]
        if not prices:
            return envelope(False, code="NOT_FOUND", message="无比价数据")
        lowest = min(prices, key=lambda x: x["price"])
        return envelope(True, data={
            "product_id": pid, "prices": prices,
            "lowest": {"platform": lowest["platform"], "price": lowest["price"]},
        })

    def _tool_authenticity_check(self, args: dict) -> dict:
        code = args.get("code")
        if not code:
            return envelope(False, code="MISSING_PARAM", message="缺少 code")
        if not AF_CODE_RE.match(str(code)):
            return envelope(False, code="INVALID_FORMAT",
                            message="防伪码格式非法(形如 AF00001042K)")
        if code not in self.grounded:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message="防伪码必须来自 OCR 结果或用户输入，禁止编造")
        rec = self.anti_fake.get(code)
        if rec is None:
            return envelope(False, code="NOT_REGISTERED",
                            message="防伪码未登记，请核对或转人工核实")
        rec["verify_count"] = rec.get("verify_count", 0) + 1
        p = self.products.get(rec["product_id"], {})
        return envelope(True, data={
            "code": code, "is_genuine": bool(rec["is_genuine"]),
            "product_title": p.get("title", ""), "verify_count": rec["verify_count"],
        })

    def _tool_query_logistics(self, args: dict) -> dict:
        order_id = args.get("order_id")
        if not order_id:
            return envelope(False, code="MISSING_PARAM", message="缺少 order_id")
        if not ORDER_ID_RE.match(str(order_id)):
            return envelope(False, code="INVALID_FORMAT",
                            message="订单号格式非法(形如 ORD00001042)")
        if order_id not in self.grounded:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message="订单号必须来自 OCR/用户输入/历史记录，禁止编造")
        track = self.logistics.get(order_id)
        if track is None:
            return envelope(False, code="NOT_FOUND",
                            message="订单号不存在，请核对后重试")
        return envelope(True, data={
            "order_id": order_id, "status": track["status"],
            "status_cn": STATUS_CN.get(track["status"], track["status"]),
            "trajectory": track["trajectory"],
        })

    def _tool_query_refund(self, args: dict) -> dict:
        order_id, refund_id = args.get("order_id"), args.get("refund_id")
        if not order_id and not refund_id:
            return envelope(False, code="MISSING_PARAM",
                            message="order_id 与 refund_id 至少提供一个")
        if refund_id:
            if refund_id not in self.grounded:
                return envelope(False, code="GROUNDING_VIOLATION",
                                message="退款单号必须来自查询结果/用户输入，禁止编造")
            r = self.refunds.get(refund_id)
            if r is None:
                return envelope(False, code="NOT_FOUND", message="退款单不存在")
        else:
            if order_id not in self.grounded:
                return envelope(False, code="GROUNDING_VIOLATION",
                                message="订单号必须来自 OCR/用户输入，禁止编造")
            rows = self.refunds_by_order.get(order_id, [])
            if not rows:
                return envelope(False, code="NOT_FOUND",
                                message=f"订单 {order_id} 无退款单")
            r = sorted(rows, key=lambda x: x["refund_id"])[-1]
        return envelope(True, data={
            "refund_id": r["refund_id"], "order_id": r["order_id"],
            "state": r["state"], "state_cn": REFUND_STATE_CN.get(r["state"], r["state"]),
            "reason": r.get("reason", ""), "amount": r.get("amount", 0),
            "updated_at": r.get("created_at", ""),
        })

    def _tool_create_refund_ticket(self, args: dict) -> dict:
        order_id = args.get("order_id")
        reason = args.get("reason")
        if not order_id or not reason:
            return envelope(False, code="MISSING_PARAM",
                            message="缺少 order_id 或 reason")
        if order_id not in self.grounded:
            return envelope(False, code="GROUNDING_VIOLATION",
                            message="订单号必须来自 OCR/用户输入/退款查询，禁止编造")
        if order_id not in self.logistics:
            return envelope(False, code="NOT_FOUND", message="订单号不存在")
        # 证据门控(PRD 06 5.8):vl_describe/ocr 证据 或 用户明确陈述
        if not (self.has_evidence or self.user_claimed_refund):
            return envelope(False, code="NO_EVIDENCE",
                            message="缺少瑕疵证据，建议先调 vl_describe/ocr 获取证据，"
                                    "或与用户确认退款理由")
        # 幂等:同订单已有进行中退款单
        active = [r for r in self.refunds_by_order.get(order_id, [])
                  if r["state"] in ACTIVE_REFUND_STATES]
        if active or order_id in self.created_refunds:
            return envelope(False, code="DUPLICATED",
                            message=f"订单 {order_id} 已有进行中的退款单")
        self.next_refund_id += 1
        refund_id = f"RF{self.next_refund_id:08d}"
        p = self.products.get(self.logistics[order_id]["product_id"], {})
        amount = args.get("amount", p.get("price", 0))
        rec = {"refund_id": refund_id, "order_id": order_id,
               "product_id": p.get("product_id"), "state": "init",
               "reason": reason, "amount": amount,
               "flaw_description": args.get("flaw_description", "")}
        self.refunds[refund_id] = rec
        self.refunds_by_order.setdefault(order_id, []).append(rec)
        self.created_refunds[order_id] = refund_id
        return envelope(True, data={
            "refund_id": refund_id, "state": "init", "created": True,
        })

    def _tool_ask_user(self, args: dict) -> dict:
        question = args.get("question")
        if not question:
            return envelope(False, code="MISSING_PARAM", message="缺少 question")
        reply = self.sim_user_replies.pop(0) if self.sim_user_replies else None
        return envelope(True, data={
            "user_reply": reply,
            "note": "等待用户回复" if reply is None else "",
        })

    def _tool_transfer_to_human(self, args: dict) -> dict:
        reason = args.get("reason")
        if not reason:
            return envelope(False, code="MISSING_PARAM", message="缺少 reason")
        self.finished = True
        return envelope(True, data={
            "handoff": True,
            "ticket": f"TK{self.rng.randint(100000, 999999)}",
            "summary": args.get("summary", "")[:200],
        })
