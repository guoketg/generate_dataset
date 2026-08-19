# 17 - 数据生成与采集

| 字段 | 值 |
|---|---|
| 文档编号 | 17 |
| 文档名称 | 数据生成与采集 |
| version | v2.0 |
| status | DRAFT |
| updated_at | 2026-08-19 |
| 负责人 | 待定 |
| 对应素材 | supplement.md（多模态电商客服主线，第 4 段差异说明）、overviwe.md 第 7、17 章（历史参考）、用户 2026-08-19 决策（H200 MIG 11 天窗口、Unsplash 官方 API、合规采集路线、本地灌库延后） |
| 关联文档 | [01-product-overview.md](01-product-overview.md)、[02-architecture.md](02-architecture.md)、[03-business-flow.md](03-business-flow.md)、[05-rag-retrieval.md](05-rag-retrieval.md)、[06-function-calling.md](06-function-calling.md)、[09-ticket-generation.md](09-ticket-generation.md)、[13-database-design.md](13-database-design.md)、[18-training-data.md](18-training-data.md)、[overview.md](overview.md)（第八节）、[../CLAUDE.md](../CLAUDE.md) |

---

## 1. 文档信息

见上表。

本文档定义多模态电商客服 Agent 的**业务平台数据资产**（商品库、比价、防伪、物流、退款、图片）的生成路径、灌库流程与资源约束。覆盖**云端一次性生成**（学校 H200 MIG，Qwen3.8-27B）与**本地灌库**（PostgreSQL + pgvector + MinIO）两阶段。**训练数据（SFT / GRPO / 评测集）不在本文档范围，见 [18-training-data.md](18-training-data.md)**；两份文档共用同一台云端 GPU 的 11 天时间窗（主时间表见 18 第 5.8 节，本文档生成阶段占 D1-D3）。

---

## 2. 背景与目标

### 2.1 背景

[03-business-flow.md](03-business-flow.md) 6.3 节规划了「自建 50 SKU + Unsplash 图」最小可用集，但实际业务场景需要更大规模、更贴近真实分布的数据：

- `image_search` / `price_compare` 需要至少 5000 SKU 才能演示 Top-K 召回效果；
- `logistics_query` 需要覆盖多状态轨迹（已发货→运输中→派送→签收 / 拒收）；
- `refund_route` 与 `refund_track_route` 需要覆盖 [09-ticket-generation.md](09-ticket-generation.md) 退款状态机的 5 种状态；
- `anti_fake` 需要真假混合的防伪码样本；
- 测试样本集需要每类业务有 4-10 张可复现的图片。

若仅依赖手工采集，按 [supplement.md](../supplement.md) 第 4 段差异说明，会陷入「跑通代码、无业务数据」的困境。本期采用 **Qwen3.8-27B 本地生成** 替代云端付费 API，避免 API 费用，同时质量接近 Claude Opus 4.6 Max（vendor 自评，详见 Qwen 模型卡）。

**关于「合规爬虫」的结论**：实时爬取淘宝 / 京东 / 亚马逊商品页不可行（登录墙 + 风控 + 合规风险，且 11 天 GPU 窗口内不值得投入）。替代路线：

- **商品元数据真实感**：优先以 McAuley Lab 学术开源数据集 **Amazon Reviews 2023 元数据**（研究合规）为种子，采样真实标题 / 类目 / 价格分布，品牌替换为虚构品牌（见 5.3.2）；下载失败则兜底纯 LLM 生成。
- **商品图片**：Unsplash 官方 API（`/search/photos` 端点，免费 Access Key，Unsplash License 免费商用），见 5.3.6。密钥经环境变量注入，**不写入仓库**。
- 平台业务库保持虚构品牌（NovaTech / Threadline 等），避免真实品牌价格误导演示环境。

### 2.2 目标

构建一套**云端一次性生成 + 本地灌库**的数据流水线，实现：

- 云端学校 H200 MIG（33 GB VRAM，暑期共用 11 天窗口，详见 [18](18-training-data.md) 5.8 主时间表）上跑 Qwen3.8-27B（4bit 量化 17-18 GB 权重，加 KV cache 实际 20-24 GB），批量生成结构化业务数据；
- 生成的数据资产以 JSONL / 图片文件形式打包（归档点 D3），下载回本地；
- 本地通过 `docker compose up` 启动 PostgreSQL+pgvector+MinIO，跑灌库脚本一键灌入（**执行延后**：本地环境就绪前不跑，见 5.4）；
- 不引入 torch / transformers 到 Python AI Agent 镜像（遵循 [CLAUDE.md](../CLAUDE.md) 第七节）；
- 所有数据资产可在本地一键复现，无需联网调 API。

### 2.3 资源约束目标

- **云端生成阶段**（一次性，预算域 C）：
  - 学校 H200 MIG 切片，33 GB VRAM；**总窗口 11 天（暑期结束归还）**，本 PRD 生成阶段占 D1-D3，其余窗口归 [18](18-training-data.md)（SFT / GRPO / 评测集）；超期兜底为自租 5090（3 元/小时）
  - Qwen3.8-27B Q4_K_M 量化约 17 GB，Q8 约 28 GB，FP8 约 28 GB；33 GB VRAM 可承载 Q4 / Q8 / FP8（H200 原生 FP8）
  - 估算 5000 SKU 生成耗时：纯 LLM 路径 Q8 单条 8-15s，约 12-20 小时；**Amazon Reviews 种子路径（推荐，见 5.3.2）LLM 仅补全 description / attributes，约 3-5 小时**
  - 生成完成后 GPU 资源切换给 18 的训练数据阶段，不重复占用
  - 与训练（qwen3.5:4b 微调）**分开跑**：27B 生成阶段与 4B 训练阶段串行，不同时加载两个模型（用户决策）
- **本地灌库阶段**（预算域 A）：
  - PostgreSQL + pgvector + MinIO 容器，总内存 ≤ 1.5 GB
  - BGE-M3 embedding 容器约 512 MB
  - 灌库 5000 SKU + 20000 比价 + 5000 防伪码 + 5000 物流轨迹，约 30 分钟
  - 图片原图 + 缩略图存储约 500 MB 磁盘
- **在线推理**（预算域 B，不变）：
  - 宿主 Ollama qwen3.5:4b 不变，3.4 GB 权重

> 关键约束：数据生成模型 Qwen3.8-27B 仅在云端使用，**不下沉到本地推理**。本地推理仍用 qwen3.5:4b，避免 27B 权重超出 8 GB VRAM 预算。

---

## 3. 名词与缩写

| 缩写 | 含义 |
|---|---|
| SKU | Stock Keeping Unit，商品最小存货单元 |
| JSONL | JSON Lines，每行一条 JSON 的文本格式 |
| Parquet | 列式存储格式，适合大数据集 |
| GGUF | llama.cpp 量化模型格式 |
| Q4_K_M | 4bit 量化方案，约 4bit/param |
| Q8 | 8bit 量化，权重精度更高 |
| FP8 | 8bit 浮点格式，vLLM/SGLang 推理使用 |
| KV Cache | Transformer 推理时的 Key-Value 缓存 |
| MinIO | S3 兼容的开源对象存储 |
| BGE-M3 | BAAI 通用嵌入模型，本项目用 1024 维 |
| 灌库 | 将生成数据批量导入数据库 |
| 冷启动 | 模型首次加载到显存的过程 |

---

## 4. 需求描述

### 4.1 功能性需求

| 编号 | 需求 |
|---|---|
| D-01 | 云端通过 Ollama 拉取 qwen3.8:27b，跑离线生成脚本，不依赖任何付费 API |
| D-02 | 生成商品主库（5000 SKU，覆盖 3C / 服饰 / 家居 / 食品 4 类目） |
| D-03 | 生成多平台比价数据（每 SKU 4 平台：jd / taobao / pdd / amazon） |
| D-04 | 生成防伪码库（每 SKU 1 个防伪码，含 5% 假货标记） |
| D-05 | 生成物流轨迹（每订单 4-6 个轨迹点，覆盖 5 种状态） |
| D-06 | 生成退款状态机测试数据（覆盖 [09](09-ticket-generation.md) 的 5 种状态） |
| D-07 | 生成测试样本集（每类业务 4-10 张图，供 e2e 测试） |
| D-08 | 用 Python PIL 合成订单截图、防伪码图、瑕疵示意图（无需 AI 绘画），数量参数化：e2e 各 20 张 + 训练数据用各 800 张（[18](18-training-data.md) 消费） |
| D-09 | 数据资产以 JSONL + 图片文件形式打包（归档点 D3），可下载回本地 |
| D-10 | 本地灌库脚本读取 JSONL，调 bge-m3 容器生成 embedding，写入 PostgreSQL+pgvector+MinIO |
| D-11 | 灌库幂等：重复执行不产生重复数据（按 product_id 去重） |
| D-12 | 商品图采集走 Unsplash 官方 API（`/search/photos`，Access Key 经环境变量 `UNSPLASH_ACCESS_KEY` 注入，密钥不入仓库） |
| D-13 | 商品库优先以 Amazon Reviews 2023 学术开源元数据为真实感种子（真实品牌映射为虚构品牌）；种子下载失败兜底纯 Qwen3.8 生成 |

### 4.2 非功能性需求

| 编号 | 维度 | 要求 |
|---|---|---|
| ND-01 | 时间 | 云端生成全部业务数据资产 ≤ 72 小时（D1-D3，H200 窗口内完成） |
| ND-02 | 数据质量 | 生成的商品标题符合电商规范（含品牌+型号+关键属性），价格分布合理 |
| ND-03 | 多样性 | 4 类目各 1250 SKU，类目内品牌/价格带分布合理 |
| ND-04 | 可复现 | 生成的 JSONL 与图片文件可重新下载，灌库结果一致 |
| ND-05 | 资源 | 云端生成阶段不占本地预算；本地灌库阶段 ≤ 2 GB RAM |
| ND-06 | 合规 | 不爬取真实电商平台页面；元数据种子来自学术开源数据集（研究用途），标题经改写、品牌替换为虚构品牌，不含真实商标与用户信息；图片走 Unsplash License（免费商用） |
| ND-07 | 可观测 | 生成脚本输出进度、耗时、失败行号 |
| ND-08 | 不引入 torch | 灌库脚本仅用 httpx + psycopg + minio + Pillow，不依赖 torch / transformers |

---

## 5. 详细设计

### 5.1 整体流程

```text
阶段 0：环境预检（D0，详见 18 5.8）
  |
  v
阶段 1：云端生成（D1-D3，预算域 C，学校 H200 MIG 33 GB）
  |
  v
云端 GPU（33 GB VRAM）
  ├── ollama pull qwen3.8:27b          # 17 GB Q4 量化
  ├── ollama serve                      # 监听 11434
  ├── scripts/fetch_amazon_seeds.py    # Amazon Reviews 2023 元数据采样（可选真实感种子）
  ├── scripts/gen_products.py          # 种子改写/兜底生成 5000 SKU + 比价 + 防伪
  ├── scripts/gen_logistics.py         # 生成物流轨迹
  ├── scripts/gen_refunds.py           # 生成退款测试数据
  ├── scripts/gen_test_samples.py      # PIL 合成订单/防伪/瑕疵图（e2e 20 + 训练 800/类）
  ├── scripts/fetch_product_images.py  # Unsplash 官方 API 拉商品图
  └── tar -czf cs_dataset.tar.gz data/   # 归档点 D3（保底资产落袋）
  |
  v
下载 cs_dataset.tar.gz 回本地（同时留在云端供 18 训练数据阶段消费）
  |
  v
阶段 2：本地灌库（预算域 A；执行延后至本地环境就绪，设计不变）
  |
  v
本地 Docker（WSL2）
  ├── docker compose up postgres minio bge-m3
  ├── scripts/seed_postgres.py         # JSONL → PostgreSQL
  ├── scripts/seed_minio.py            # 图片 → MinIO
  └── scripts/seed_embedding.py        # 调 bge-m3 容器 → pgvector
  |
  v
验证：跑 6 大业务路由 e2e 测试
```

### 5.2 数据资产清单

| 资产 | 表 / 文件 | 规模 | 用途 | 生成方式 |
|---|---|---|---|---|
| 商品主库 | cs_product | 5000 行 | image_search / price_compare / consult_route | Amazon Reviews 2023 种子改写（推荐）/ Qwen3.8 兜底生成 |
| 多平台比价 | cs_product_price | 20000 行 | price_compare | 规则 + 随机扰动 |
| 防伪码库 | cs_anti_fake_code | 5000 行（5% 假） | anti_fake | 规则生成 |
| 物流轨迹 | cs_logistics_track | 5000 单 × 4-6 点 | logistics_query | 模板 + 随机状态 |
| 退款测试数据 | cs_refund_ticket | 500 行（5 状态） | refund_route / refund_track_route | 状态机采样 |
| 商品图 | data/images/products/*.jpg | 5000 张 | image_search 输入 | Unsplash 官方 API（/search/photos，每 SKU 1 张） |
| 订单截图 | data/images/orders/*.png | 20 张（e2e）+ 800 张（训练） | OCR + logistics + refund_field_extract | PIL 合成 |
| 防伪码图 | data/images/anti_fake/*.png | 20 张（e2e）+ 800 张（训练） | anti_fake 演示 | PIL + qrcode 合成 |
| 瑕疵示意图 | data/images/defects/*.jpg | 20 张（e2e）+ 800 张（训练） | refund_route 瑕疵判定 | Unsplash 商品图 + PIL 标注红框 |
| 测试样本集 | test/samples/*.jpg | 20 张 | 6 大业务 e2e 测试 | 上述各类各取 4 张 |

> 「训练」数量的合成图与商品 JSONL 在云端留存（tar 归档 #1），供 [18](18-training-data.md) 构造 SFT / GRPO / 评测集时直接消费（训练态工具执行器读同一套资产，见 [06](06-function-calling.md) 5.10）。

### 5.3 云端生成阶段详设

#### 5.3.1 环境准备

```bash
# 云端 GPU 机器（学校 H200 MIG，33 GB VRAM）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.8:27b          # Q4_K_M，约 17 GB 下载
ollama serve                      # 后台监听 11434

# 网络预检（D0 检查项，详见 18 5.8）：任一失败走 7.2 异常预案
curl -sI https://api.unsplash.com/photos/random?client_id=$UNSPLASH_ACCESS_KEY | head -1   # Unsplash 可达
curl -sI https://hf-mirror.com | head -1                                                  # HF 镜像可达（公开 QA 下载走 18）
# Amazon Reviews 2023 元数据可达性（McAuley Lab / 镜像源）

# 验证
ollama list                       # 应见 qwen3.8:27b
curl http://localhost:11434/api/tags
```

> 密钥通过环境变量注入（`UNSPLASH_ACCESS_KEY`），配置在云端机器的 `~/.bashrc` 或 `.env`（不入仓库、不入 PRD）。

#### 5.3.2 商品库生成（Amazon Reviews 2023 种子 + Qwen3.8 兜底）

**推荐路径（种子改写，约 3-5 小时）**：以 McAuley Lab「Amazon Reviews 2023」数据集的**元数据文件**（title / categories / price，研究合规）为真实感种子，替代纯 LLM 凭空生成，避免 27B 批量生成常见的模式重复：

```text
Amazon Reviews 2023 元数据（按类目文件流式采样，不存全量）
  类目映射：Electronics→3c / Clothing_Shoes_and_Jewelry→clothing
           / Home_and_Kitchen→home / Grocery_and_Gourmet_Food→food
        |
        v
每类目采样 ~1300 条有效记录（含 title / price / category）
        |
        v
规则改写（无 GPU 消耗，秒级）：
  ├── title：保留型号/属性词，真实品牌词 → 虚构品牌（BRANDS 映射表）
  ├── price：真实价格带校准（异常值剔除，按类目分位数裁剪）
  └── brand/model/category：映射与抽取
        |
        v
Qwen3.8-27B 仅补全 description + attributes（批量 50 条/次，~1-2 小时）
        |
        v
products.jsonl（5000 行）
```

种子采样的关键点：

| 项 | 约束 |
|---|---|
| 采样量 | 每类目 1300 条（冗余 4% 供过滤），流式解析，不全量下载 |
| 品牌替换 | 正则抽取真实品牌词 → 按类目映射到虚构品牌表（BRANDS），保证标题自然 |
| 价格分布 | 保留真实分布形态（长尾），仅裁剪类目 1%/99% 分位外的异常值 |
| 去重 | 标题归一化（去品牌/空白）后近似去重，避免同款刷屏 |
| 兜底 | 种子下载 / 解析失败 → 走下方纯 LLM 生成路径（v1.0 原方案） |

**兜底路径（纯 LLM 生成，约 12-20 小时）**：以下为 v1.0 原脚本，作为种子不可用时的 fallback：

```python
# scripts/gen_products.py（云端运行，不入 Docker）
"""
生成 5000 SKU 商品主库 + 多平台比价 + 防伪码
输出：data/products.jsonl / prices.jsonl / anti_fake.jsonl
"""
import httpx, json, time, random
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.8:27b"

CATEGORIES = {
    "3c": ["smartphone", "laptop", "headphone", "tablet", "camera"],
    "clothing": ["t-shirt", "sneaker", "jacket", "dress", "bag"],
    "home": ["mug", "lamp", "sofa", "knife", "vase"],
    "food": ["coffee", "snack", "fruit", "tea", "chocolate"],
}
BRANDS = {  # 每类目 5-10 个品牌，避免真实商标
    "3c": ["NovaTech", "Pulse", "OrbitX", "VertexQ", "LumenA"],
    "clothing": ["Threadline", "VogueStep", "AuroraWear", "CobaltBay", "MapleCo"],
    "home": ["HearthHome", "Lumio", "CedarWorks", "PaleMoon", "TideStudio"],
    "food": ["BeanVista", "SnackHive", "OrchardGold", "SteepLeaf", "CocoaRidge"],
}
PLATFORMS = ["jd", "taobao", "pdd", "amazon"]

PROMPT_TEMPLATE = """You are an e-commerce product data generator. Output strict JSON only, no markdown.
Generate {n} unique products in category "{cat}" with brands from: {brands}.
Each product object:
{{
  "title": "<Brand> <Model> <KeyAttr> <CategoryNoun>, <=30 chars Chinese/English",
  "brand": "<one of given brands>",
  "model": "<short alphanumeric model code, e.g. NX-4501>",
  "category": "{cat}",
  "price": <number, 9.9 ~ 9999.0, reasonable for category>,
  "description": "<<=50 chars, mention 1-2 selling points>",
  "attributes": {{"color": "...", "size": "...", "material": "..."}}
}}
Output as JSON array only, no other text. Do not repeat titles."""

def call_qwen(prompt: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        r = httpx.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": False,
            "options": {"temperature": 0.9, "num_ctx": 8192, "num_predict": 4096},
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=300)
        try:
            content = r.json()["message"]["content"]
            # Qwen3.8 默认带 thinking，需提取 <answer>...</answer> 或最后一个 JSON 数组
            return extract_json_array(content)
        except Exception as e:
            print(f"attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    return []

def extract_json_array(text: str) -> list[dict]:
    """Qwen3.8 thinking 模式输出可能含 <think>...</think> 与 JSON，需提取"""
    import re
    # 移除 <think>...</think> 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 找最后一个 [ ... ] 块
    matches = re.findall(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if not matches:
        return []
    return json.loads(matches[-1])

def gen_products():
    Path("data").mkdir(exist_ok=True)
    products, prices, anti_fake = [], [], []
    fake_indices = set(random.sample(range(5000), 250))  # 5% 假货

    for cat, nouns in CATEGORIES.items():
        for noun in nouns:
            prompt = PROMPT_TEMPLATE.format(
                n=250, cat=cat, brands=BRANDS[cat]
            )
            items = call_qwen(prompt)
            for idx, item in enumerate(items):
                pid = len(products) + 1
                product = {
                    "product_id": pid,
                    "title": item.get("title", f"{cat}_{pid}")[:60],
                    "category": cat,
                    "brand": item.get("brand", "Unknown"),
                    "price": float(item.get("price", 99.0)),
                    "platform": random.choice(PLATFORMS),
                    "attributes": item.get("attributes", {}),
                }
                products.append(product)
                # 多平台比价（基准价 ± 15%）
                for p in PLATFORMS:
                    prices.append({
                        "product_id": pid,
                        "platform": p,
                        "price": round(product["price"] * random.uniform(0.85, 1.15), 2),
                    })
                # 防伪码（5% 假）
                anti_fake.append({
                    "code": f"AF{pid:08d}{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}",
                    "product_id": pid,
                    "is_genuine": pid not in fake_indices,
                })
            print(f"cat={cat} noun={noun} done, total={len(products)}")
            if len(products) >= 5000:
                break
        if len(products) >= 5000:
            break

    with open("data/products.jsonl", "w", encoding="utf-8") as f:
        for p in products[:5000]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open("data/prices.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in prices[:20000])
    with open("data/anti_fake.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in anti_fake[:5000])

if __name__ == "__main__":
    gen_products()
```

> Qwen3.8-27B 默认 `reasoning_effort=xhigh`，对简单数据生成过度思考，需在 `options` 中显式关闭 thinking 或降低 reasoning_effort，否则单条耗时翻倍。

#### 5.3.3 物流轨迹生成

```python
# scripts/gen_logistics.py
"""依赖 data/products.jsonl，为每 SKU 生成 1 个订单的物流轨迹"""
import json, random
from pathlib import Path

STATUSES = ["shipped", "in_transit", "delivering", "signed", "rejected"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]

def gen_track(order_id: str, product_id: int) -> dict:
    n_points = random.randint(4, 6)
    status_seq = STATUSES[:n_points-1] + [random.choice(["signed", "rejected"])]
    trajectory = []
    for i, st in enumerate(status_seq):
        trajectory.append({
            "ts": f"2026-08-{10+i:02d} {10+i:02d}:30:00",
            "location": random.choice(CITIES),
            "action": st,
        })
    return {
        "order_id": order_id,
        "product_id": product_id,
        "status": status_seq[-1],
        "trajectory": trajectory,
    }

def main():
    tracks = []
    with open("data/products.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            tracks.append(gen_track(f"ORD{p['product_id']:08d}", p["product_id"]))
    with open("data/logistics.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in tracks)

if __name__ == "__main__":
    main()
```

#### 5.3.4 退款状态机测试数据

```python
# scripts/gen_refunds.py
"""覆盖 09-ticket-generation.md 退款状态机 5 种状态"""
import json, random

REFUND_STATES = ["init", "reviewing", "approved", "refunded", "rejected"]
REFUND_REASONS = {
    "init": "用户发起退款申请，等待审核",
    "reviewing": "客服审核中，需补充瑕疵图",
    "approved": "审核通过，等待退款到账",
    "refunded": "退款已到账，订单关闭",
    "rejected": "退款被拒，原因：商品无瑕疵",
}

def gen_refunds(n_per_state: int = 100):
    refunds = []
    rid = 1
    for state in REFUND_STATES:
        for _ in range(n_per_state):
            pid = random.randint(1, 5000)
            refunds.append({
                "refund_id": f"RF{rid:08d}",
                "order_id": f"ORD{pid:08d}",
                "product_id": pid,
                "state": state,
                "reason": REFUND_REASONS[state],
                "amount": round(random.uniform(20, 2000), 2),
                "created_at": f"2026-08-{random.randint(1,18):02d} {random.randint(8,22):02d}:00:00",
            })
            rid += 1
    with open("data/refunds.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in refunds)

if __name__ == "__main__":
    gen_refunds()
```

#### 5.3.5 图片合成（PIL，不依赖 AI 绘画）

```python
# scripts/gen_test_samples.py
"""
合成订单截图 / 防伪码图 / 瑕疵标注图
所有图片走 PIL 合成，不调任何文生图 API
数量参数化：--n-e2e 20（e2e 测试）--n-train 800（供 18 训练数据消费）
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import qrcode, json, random
from pathlib import Path

def gen_order_screenshot(order_id: str, product_title: str, price: float, n: int):
    """订单截图模板，含订单号、商品、价格、物流状态"""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        big_font = ImageFont.truetype("arialbd.ttf", 28)
    except:
        font = ImageFont.load_default()
        big_font = font
    draw.text((20, 20), "订单详情", fill="black", font=big_font)
    draw.text((20, 80), f"订单号：{order_id}", fill="black", font=font)
    draw.text((20, 120), f"商品：{product_title}", fill="black", font=font)
    draw.text((20, 160), f"金额：￥{price:.2f}", fill="red", font=font)
    draw.text((20, 200), "状态：运输中", fill="blue", font=font)
    img.save(f"data/images/orders/order_{n}.png")

def gen_anti_fake_image(code: str, product_id: int, n: int):
    """防伪码图：二维码 + 文字标签"""
    qr = qrcode.make(code).resize((300, 300))
    img = Image.new("RGB", (400, 400), "white")
    img.paste(qr, (50, 20))
    draw = ImageDraw.Draw(img)
    draw.text((50, 340), f"防伪码：{code}", fill="black")
    draw.text((50, 370), f"商品 ID：{product_id}", fill="black")
    img.save(f"data/images/anti_fake/af_{n}.png")

def gen_defect_image(product_img_path: str, n: int):
    """瑕疵图：商品图 + 红框标注瑕疵位置"""
    img = Image.open(product_img_path).convert("RGB").resize((400, 400))
    draw = ImageDraw.Draw(img)
    x, y = random.randint(50, 300), random.randint(50, 300)
    draw.rectangle([x, y, x+80, y+80], outline="red", width=4)
    img.save(f"data/images/defects/defect_{n}.jpg")

def main():
    for d in ["orders", "anti_fake", "defects"]:
        Path(f"data/images/{d}").mkdir(parents=True, exist_ok=True)
    with open("data/products.jsonl", encoding="utf-8") as f:
        products = [json.loads(l) for l in f]
    # 订单截图：e2e 20 张 + 训练 800 张（参数化，同函数不同 n）
    for i in range(N_E2E + N_TRAIN):
        p = random.choice(products)
        gen_order_screenshot(f"ORD{p['product_id']:08d}", p["title"], p["price"], i)
    # 防伪码图：e2e 20 张 + 训练 800 张
    with open("data/anti_fake.jsonl", encoding="utf-8") as f:
        afs = [json.loads(l) for l in f]
    for i in range(N_E2E + N_TRAIN):
        af = random.choice(afs)
        gen_anti_fake_image(af["code"], af["product_id"], i)
    # 瑕疵图：e2e 20 张 + 训练 800 张（基于商品图加红框）
    product_imgs = list(Path("data/images/products").glob("*.jpg"))
    for i in range(min(N_E2E + N_TRAIN, len(product_imgs))):
        gen_defect_image(str(product_imgs[i]), i)

if __name__ == "__main__":
    main()
```

#### 5.3.6 商品图采集（Unsplash 官方 API，真实图片）

用户已提供 Unsplash Access Key（存云端 `.env`，环境变量 `UNSPLASH_ACCESS_KEY`，**密钥不写入仓库与文档**）。`source.unsplash.com` 已废弃（v1.0 方案作废），改走官方 `/search/photos` 端点：

| 项 | 值 |
|---|---|
| 端点 | `GET https://api.unsplash.com/search/photos?query={q}&per_page=30&page={p}&client_id={ACCESS_KEY}` |
| 配额 | Demo 档 50 请求/小时（仅计搜索请求；`images.unsplash.com` 图片下载不限该配额） |
| 请求量估算 | 5000 张 ÷ 30 张/页 ≈ 167 次搜索 ≈ 3.5 小时（配额自动节流） |
| License | Unsplash License，免费商用、无需署名 |
| 尺寸 | 取 `urls.regular`（宽 1080px），满足 [03](03-business-flow.md) ≤ 2048px 预处理约束 |

```python
# scripts/fetch_product_images.py
"""Unsplash 官方 API 拉取 5000 张真实商品图（断点续传 + 配额节流）"""
import httpx, asyncio, os, time, json
from pathlib import Path

ACCESS_KEY = os.environ["UNSPLASH_ACCESS_KEY"]   # 不入库，云端 .env 注入
SEARCH_URL = "https://api.unsplash.com/search/photos"
QUERIES = {
    "3c": ["smartphone", "laptop", "headphone", "tablet", "camera"],
    "clothing": ["t-shirt", "sneaker", "jacket", "dress", "bag"],
    "home": ["mug", "lamp", "sofa", "knife", "vase"],
    "food": ["coffee", "snack", "fruit", "tea", "chocolate"],
}
STATE = Path("data/images/products/.fetch_state.json")  # 断点续传游标

async def search_page(client: httpx.AsyncClient, query: str, page: int) -> list:
    r = await client.get(SEARCH_URL, params={
        "query": query, "per_page": 30, "page": page, "client_id": ACCESS_KEY,
    }, timeout=30)
    if r.status_code == 429:                    # 配额用尽：等下一小时窗口
        time.sleep(3600 - (time.time() % 3600) + 5)
        return await search_page(client, query, page)
    r.raise_for_status()
    return [p["urls"]["regular"] for p in r.json()["results"]]

async def download(client: httpx.AsyncClient, url: str, path: Path):
    if path.exists():                           # 断点续传：已下载跳过
        return
    r = await client.get(url, timeout=60)
    if r.status_code == 200:
        path.write_bytes(r.content)

async def main():
    out = Path("data/images/products"); out.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    async with httpx.AsyncClient() as client:
        for cat, queries in QUERIES.items():
            for q in queries:
                got = state.get(f"{cat}/{q}", 0)            # 每词需 250 张 ≈ 9 页
                while got < 250:
                    page = got // 30 + 1
                    urls = await search_page(client, q, page)
                    if not urls:
                        break                                # 该词结果不足，记录后换词
                    for j, u in enumerate(urls):
                        await download(client, u, out / f"{cat}_{q}_{got+j:04d}.jpg")
                    got += len(urls)
                    state[f"{cat}/{q}"] = got
                    STATE.write_text(json.dumps(state))      # 每页落盘游标

if __name__ == "__main__":
    asyncio.run(main())
```

> 下载不占 GPU，可与 5.3.2 商品库生成并行跑（CPU/IO 任务）。若单查询结果不足 250 张（冷门词），由同类目其他词补足；整体缺口 > 10% 时按 7.2 预案补拉或对缺失 SKU 复用同类目图片（同一商品多视角不追求，单图即可）。

### 5.4 本地灌库阶段详设

> **执行延后**（用户 2026-08-19 指示）：本地环境（CUDA / Docker 资源）就绪前，本阶段设计与脚本仅归档，不执行——避免占用本地 CPU / 内存。云端阶段（5.3）产物先行打包落袋（归档点 D3）。

#### 5.4.1 灌库脚本架构

```python
# scripts/seed_all.py（本地运行，依赖 docker compose 已起 postgres/minio/bge-m3）
"""
读取 data/*.jsonl 与 data/images/*，灌入：
- PostgreSQL：cs_product / cs_product_price / cs_anti_fake_code / cs_logistics_track / cs_refund_ticket
- MinIO：data/images/products/*.jpg → bucket=cs-products
- pgvector：调 bge-m3 容器对每个商品 title 生成 embedding
"""
import httpx, psycopg, minio, json, io
from pathlib import Path
from PIL import Image

PG_DSN = "postgresql://cs:cs@localhost:5432/cs_agent"
MINIO_ENDPOINT = "localhost:9000"
BGE_URL = "http://localhost:8081/embed"  # bge-m3 容器

def make_thumbnail(img_bytes: bytes, max_size: int = 256) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()  # ≤ 16 KB

def embed(text: str) -> list[float]:
    r = httpx.post(BGE_URL, json={"inputs": [text]}, timeout=30)
    return r.json()[0]  # 1024 维

def seed_products():
    mc = minio.Minio(MINIO_ENDPOINT, "minioadmin", "minioadmin", secure=False)
    if not mc.bucket_exists("cs-products"):
        mc.make_bucket("cs-products")
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        with open("data/products.jsonl", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                # 上传图片到 MinIO
                img_path = f"data/images/products/{p['category']}_{p['product_id']}.jpg"
                if Path(img_path).exists():
                    img_bytes = Path(img_path).read_bytes()
                    obj = f"products/{p['product_id']}.jpg"
                    mc.put_object("cs-products", obj, io.BytesIO(img_bytes), len(img_bytes), "image/jpeg")
                    thumb = make_thumbnail(img_bytes)
                    mc.put_object("cs-products", f"thumb/{obj}", io.BytesIO(thumb), len(thumb), "image/jpeg")
                    image_url = f"http://localhost:9000/cs-products/{obj}"
                    thumb_url = f"http://localhost:9000/cs-products/thumb/{obj}"
                else:
                    image_url = thumb_url = ""
                # 写 PostgreSQL（幂等：ON CONFLICT 跳过）
                cur.execute("""
                    INSERT INTO cs_product (product_id, title, category, brand, price, platform, image_url, thumb_url, attributes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (product_id) DO NOTHING
                """, (p["product_id"], p["title"], p["category"], p["brand"], p["price"],
                      p["platform"], image_url, thumb_url, p.get("attributes", {})))
                # 生成 embedding 写入 pgvector
                emb = embed(p["title"])
                cur.execute("""
                    INSERT INTO cs_product_embedding (product_id, embedding)
                    VALUES (%s, %s) ON CONFLICT (product_id) DO NOTHING
                """, (p["product_id"], emb))
        conn.commit()

def seed_prices():
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        with open("data/prices.jsonl", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                cur.execute("""
                    INSERT INTO cs_product_price (product_id, platform, price)
                    VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
                """, (p["product_id"], p["platform"], p["price"]))
        conn.commit()

# seed_anti_fake / seed_logistics / seed_refunds 同结构...

if __name__ == "__main__":
    seed_products()
    seed_prices()
    # ...
    print("seed done")
```

#### 5.4.2 Docker Compose 增项

新增 MinIO 与 bge-m3 容器到 `docker/docker-compose.all.yml`：

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: cs-postgres
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: cs
      POSTGRES_PASSWORD: cs
      POSTGRES_DB: cs_agent
    volumes:
      - ./volumes/postgres/data:/var/lib/postgresql/data
      - ./init/products.sql:/docker-entrypoint-initdb.d/02-products.sql
    mem_limit: 1g
    networks: [cs-net]

  minio:
    image: minio/minio:RELEASE.2024-10-13T13-34-07Z  # 固定 tag
    container_name: cs-minio
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes: [./volumes/minio/data:/data]
    mem_limit: 256m
    networks: [cs-net]

  bge-m3:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
    container_name: cs-bge-m3
    ports: ["8081:80"]
    environment:
      MODEL_ID: BAAI/bge-m3
      INT8: "1"
    volumes: [./volumes/hf-cache:/data]
    mem_limit: 512m
    networks: [cs-net]

networks:
  cs-net:
```

#### 5.4.3 数据库初始化 SQL

`docker/init/products.sql`（容器启动时自动执行）：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS cs_product (
    product_id   BIGSERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    category     VARCHAR(64) NOT NULL,
    brand        VARCHAR(64),
    price        NUMERIC(10,2) NOT NULL,
    platform     VARCHAR(32) NOT NULL,
    image_url    TEXT,
    thumb_url    TEXT,
    attributes   JSONB DEFAULT '{}'::jsonb,
    created_at   TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_product_category ON cs_product(category);
CREATE INDEX IF NOT EXISTS idx_product_title_trgm ON cs_product USING gin(title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS cs_product_price (
    id           BIGSERIAL PRIMARY KEY,
    product_id   BIGINT REFERENCES cs_product(product_id),
    platform     VARCHAR(32) NOT NULL,
    price        NUMERIC(10,2) NOT NULL,
    url          TEXT,
    updated_at   TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cs_product_embedding (
    product_id   BIGINT PRIMARY KEY REFERENCES cs_product(product_id),
    embedding    vector(1024) NOT NULL,
    model         VARCHAR(32) DEFAULT 'bge-m3'
);
CREATE INDEX IF NOT EXISTS idx_product_embedding_hnsw
  ON cs_product_embedding USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS cs_anti_fake_code (
    code         VARCHAR(64) PRIMARY KEY,
    product_id   BIGINT REFERENCES cs_product(product_id),
    is_genuine   BOOLEAN DEFAULT TRUE,
    verify_count INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cs_logistics_track (
    order_id     VARCHAR(32) PRIMARY KEY,
    product_id   BIGINT REFERENCES cs_product(product_id),
    status       VARCHAR(32),
    trajectory   JSONB,
    updated_at   TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cs_refund_ticket (
    refund_id    VARCHAR(32) PRIMARY KEY,
    order_id     VARCHAR(32) NOT NULL,
    product_id   BIGINT REFERENCES cs_product(product_id),
    state        VARCHAR(32) NOT NULL,
    reason       TEXT,
    amount       NUMERIC(10,2),
    created_at   TIMESTAMP
);
```

### 5.5 资源预算归属

按 [overview.md](overview.md) 第八节三预算域：

| 项目 | 预算域 | 说明 |
|---|---|---|
| 云端 GPU 跑 Qwen3.8-27B 生成 | **C（云端）** | 一次性，不占本地 |
| 云端下载的 JSONL/图片包 | 磁盘 | 本地仓库，约 500 MB |
| PostgreSQL + pgvector（灌库后） | **A** | 约 200 MB |
| MinIO（5000 商品图 + 缩略图） | **A** | 约 250 MB 磁盘 + 256 MB RAM |
| BGE-M3 容器 | **A** | 512 MB RAM |
| 本地推理 qwen3.5:4b | **B** | 3.4 GB 权重，不变 |

> 不新增预算域。云端生成属一次性任务，归入 C；本地灌库产物归入 A；本地推理不变。

### 5.6 验证流程

```text
1. docker compose up -d postgres minio bge-m3
2. 等待 bge-m3 健康检查通过（/health 端点）
3. python scripts/seed_all.py
4. 验证：
   - psql -c "SELECT count(*) FROM cs_product" → 5000
   - psql -c "SELECT count(*) FROM cs_product_embedding" → 5000
   - curl http://localhost:9000/cs-products/products/1.jpg → 图片可下载
   - python test/run_e2e.py → 6 大业务路由全部跑通
```

---

## 6. 数据与接口

### 6.1 数据资产契约

| 文件 | Schema | 规模 | 行示例 |
|---|---|---|---|
| data/products.jsonl | 见 5.3.2 | 5000 行 | `{"product_id":1,"title":"NovaTech NX-4501...","category":"3c",...}` |
| data/prices.jsonl | 见 5.3.2 | 20000 行 | `{"product_id":1,"platform":"jd","price":299.99}` |
| data/anti_fake.jsonl | 见 5.3.2 | 5000 行 | `{"code":"AF00000001A","product_id":1,"is_genuine":true}` |
| data/logistics.jsonl | 见 5.3.3 | 5000 行 | `{"order_id":"ORD00000001","product_id":1,"status":"signed",...}` |
| data/refunds.jsonl | 见 5.3.4 | 500 行 | `{"refund_id":"RF00000001","order_id":"ORD00000001",...}` |
| data/images/products/*.jpg | 见 5.3.6 | 5000 张 | 商品正面图（Unsplash 真实图） |
| data/images/orders/*.png | 见 5.3.5 | 20 张（e2e）+ 800 张（训练） | 订单截图 |
| data/images/anti_fake/*.png | 见 5.3.5 | 20 张（e2e）+ 800 张（训练） | 防伪码图 |
| data/images/defects/*.jpg | 见 5.3.5 | 20 张（e2e）+ 800 张（训练） | 瑕疵标注图 |

### 6.2 接口契约

| 接口 | 方法 | 用途 |
|---|---|---|
| `http://localhost:11434/api/chat` | POST | 调 Qwen3.8-27B 生成（云端阶段） |
| `https://api.unsplash.com/search/photos` | GET | 商品图搜索（云端阶段，`UNSPLASH_ACCESS_KEY` 环境变量注入） |
| Amazon Reviews 2023 元数据文件 | 下载 | 商品库真实感种子（云端阶段，流式采样） |
| `http://localhost:8081/embed` | POST | 调 bge-m3 容器生成 embedding（本地阶段） |
| `http://localhost:9000` | S3 API | MinIO 对象存储（本地阶段） |
| `postgresql://cs:cs@localhost:5432/cs_agent` | TCP | PostgreSQL 数据库（本地阶段） |

---

## 7. 边界与异常

### 7.1 流程边界

- 本期做 5000 SKU 静态数据集，不做实时电商 API 对接，**不爬取真实电商平台页面**（合规结论见 2.1）。
- 本期做合成瑕疵图（PIL 红框标注），不做真实瑕疵图采集。
- 本期不做图片文生图（不调 SD/DALL-E/Midjourney）。
- 本期不做训练数据集生成（SFT/GRPO/评测集见 [18-training-data.md](18-training-data.md)，与本文档共用 H200 时间窗与数据资产）。
- 本地灌库执行延后（用户指示），云端产物先行归档。

### 7.2 异常处理

| 异常 | 处理 |
|---|---|
| Qwen3.8 输出非 JSON | `extract_json_array` 兜底提取，失败重试 3 次 |
| Qwen3.8 thinking 模式拖慢 | `options` 显式降低 reasoning_effort 或 disable thinking |
| Unsplash API 429（配额用尽） | 脚本自动等待下一小时窗口（5.3.6 内置节流）；整体缺口 > 10% 时同类目图片复用补位 |
| Unsplash API 在学校网络不可达 | 本地拉取后 scp 上传云端（图片下载不占 GPU/CPU 大头）；或对缺失 SKU 复用同类目图 |
| Amazon Reviews 2023 种子下载/解析失败 | 兜底走纯 Qwen3.8 生成路径（v1.0 原方案，多耗 8-15 小时） |
| H200 窗口超支（D3 未完成） | 优先级保产物：products/prices/anti_fake 必产（分钟-小时级）；图片与合成图可延后补（18 阶段前补齐即可） |
| bge-m3 容器冷启动慢 | 灌库脚本启动后 sleep 30s 等待健康检查 |
| MinIO bucket 不存在 | 灌库脚本启动时 `make_bucket` 幂等创建 |
| PostgreSQL 连接拒绝 | docker compose ps 检查容器状态，等待 ready 后重试 |
| 图片下载失败 | 跳过该 SKU，记录缺失日志，image_url 留空 |
| 灌库中断 | 脚本幂等（ON CONFLICT DO NOTHING），可重跑 |

### 7.3 资源约束（强制）

- 云端生成阶段：33 GB VRAM GPU，Qwen3.8-27B Q4_K_M ≤ 24 GB（含 KV cache），单实例串行。
- 本地灌库阶段：Docker/WSL ≤ 8 GB RAM，PostgreSQL+MinIO+bge-m3 总内存 ≤ 1.5 GB。
- 本地推理不变：qwen3.5:4b 3.4 GB 权重，不动。
- 详细约束见 [overview.md](overview.md) 第八节与 [../CLAUDE.md](../CLAUDE.md) 第五节。

---

## 8. 验收标准

- [ ] 云端 `ollama pull qwen3.8:27b` 成功，`ollama list` 可见。
- [ ] 云端跑完 `gen_products.py`（种子或兜底路径）输出 5000 行 `products.jsonl`，无 JSON 解析失败；品牌全部为虚构品牌表内值。
- [ ] `gen_logistics.py` / `gen_refunds.py` / `gen_test_samples.py` 全部产出预期文件（合成图 e2e 20/类 + 训练 800/类）。
- [ ] `fetch_product_images.py` 经官方 API 下载 ≥ 4500 张真实商品图（允许 10% 失败，断点续传可重跑）。
- [ ] D3 归档点：`cs_dataset.tar.gz` 打包完成（体积 ≤ 1 GB，不含训练用合成图时）并回传本地一份。
- [ ] **本地灌库验收项延后执行**（环境就绪后）：`docker compose up -d postgres minio bge-m3` 三容器健康；`seed_all.py` 后 `cs_product` = 5000、`cs_product_embedding` = 5000（1024 维）；MinIO bucket 含 5000 原图 + 5000 缩略图；6 大业务路由 e2e 通过。
- [ ] 全流程不引入 torch / transformers 到 Python AI Agent 镜像。
- [ ] 全流程无付费 API 调用、无真实电商平台爬取。

---

## 9. 关联文档

- [01-product-overview.md](01-product-overview.md)：训练闭环与数据规划（训练数据落地见 18）。
- [02-architecture.md](02-architecture.md)：在线 + 离线架构、三预算域。
- [03-business-flow.md](03-business-flow.md)：6.3 节商品库与图片数据来源（本文档扩展）。
- [05-rag-retrieval.md](05-rag-retrieval.md)：RAG 五级检索，依赖 cs_product + cs_product_embedding。
- [06-function-calling.md](06-function-calling.md)：11 工具契约；其训练态执行器（5.10 节）消费本文档数据资产。
- [09-ticket-generation.md](09-ticket-generation.md)：退款状态机，本文档生成其测试数据。
- [13-database-design.md](13-database-design.md)：完整数据库 schema（待撰写，本文档给出 cs_* 表子集）。
- [18-training-data.md](18-training-data.md)：训练数据生成（SFT/GRPO/评测集），与本文档共用 H200 时间窗（本文档占 D1-D3）。
- [overview.md](overview.md)：第八节内存占用分析与三预算域。
- [../CLAUDE.md](../CLAUDE.md)：第五节资源约束、第七节禁止事项。
- [../supplement.md](../supplement.md)：第 4 段差异说明（开源数据 vs 业务数据）。

---

## 10. 变更记录

| 版本 | 日期 | 变更人 | 变更说明 |
|---|---|---|---|
| v1.0 | 2026-08-19 | - | 初始版本。规划云端 Qwen3.8-27B 一次性生成 + 本地灌库两阶段流水线；定义 5000 SKU / 20000 比价 / 5000 防伪 / 5000 物流 / 500 退款测试数据资产清单；给出 gen_*.py / seed_all.py 脚本架构；新增 MinIO + bge-m3 docker-compose 增项；明确预算域归属（C 云端生成，A 本地灌库，B 推理不变）。 |
| v2.0 | 2026-08-19 | - | 对齐用户 2026-08-19 决策：GPU 定为学校 H200 MIG（33 GB，11 天窗口，生成阶段占 D1-D3，超期兜底自租 5090）；商品图采集由 source.unsplash.com（已废弃）改为 Unsplash 官方 /search/photos API（Access Key 走环境变量，断点续传 + 配额节流）；新增 Amazon Reviews 2023 学术开源元数据作为商品库真实感种子（品牌替换为虚构品牌，纯 LLM 生成降为兜底，生成耗时 12-20h → 3-5h）；明确「不爬取真实电商平台页面」合规结论；合成图数量参数化（e2e 20/类 + 训练 800/类，供 18 消费）；本地灌库执行延后（避免占用本地 CPU/内存），设计不变；新增 D3 归档点（保底资产落袋）。 | |
