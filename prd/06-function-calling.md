# 06 - Function Calling 与工具体系

| 字段 | 值 |
|---|---|
| 文档编号 | 06 |
| 文档名称 | Function Calling 与工具体系 |
| version | v1.0 |
| status | DRAFT |
| updated_at | 2026-08-19 |
| 负责人 | 待定 |
| 对应素材 | supplement.md（8 工具代码现状 + 23 工具工业方案）、answer/data.md（轨迹格式选型结论：OpenAI Tool Call JSON） |
| 关联文档 | [01-product-overview.md](01-product-overview.md)、[02-architecture.md](02-architecture.md)、[03-business-flow.md](03-business-flow.md)、[04-intent-routing.md](04-intent-routing.md)、[05-rag-retrieval.md](05-rag-retrieval.md)、[07-model-fallback.md](07-model-fallback.md)、[08-quality-and-human.md](08-quality-and-human.md)、[09-ticket-generation.md](09-ticket-generation.md)、[17-data-generation.md](17-data-generation.md)、[overview.md](overview.md)（第八节）、[../CLAUDE.md](../CLAUDE.md) |

---

## 1. 文档信息

见上表。

本文档定义多模态电商客服 Agent 的**工具调用协议（OpenAI Tool Call JSON）与工具体系（11 个工具）**，并给出每个工具的 JSON Schema 定稿、参数溯源约束、各业务路由参考工具链、训练轨迹数据格式。

> **本文档是 SFT / GRPO 训练数据生成（[01](01-product-overview.md) 5.4 节、[17](17-data-generation.md)）与在线 LangGraph 工具调用（[03](03-business-flow.md) tool_call_node）的硬前置依赖**：没有定稿的工具 Schema，就无法构造轨迹训练数据，也无法计算格式 / 过程 / 成本三类 Reward。工具 Schema 单一契约源为 `config/tools/tools.json`，全部消费方（在线服务、Teacher 轨迹生成、SFT 数据集、GRPO Rollout 环境）只读该文件。

---

## 2. 背景与目标

### 2.1 背景

工具体系存在两版历史口径，需要在本文档一次性收敛：

| 口径 | 来源 | 工具数 | 问题 |
|---|---|---|---|
| 工业方案 | supplement.md 面试文档 / [01](01-product-overview.md) 5.3 节 | 23 | 规模过大：模型面对大工具表选择困惑，即使强模型也难以稳定选对；6 个图像预处理工具属确定性操作，不需要模型决策 |
| 代码现状 | supplement.md 代码 | 8 | crop / layout_parsing / text_search / web_search / image_search / perspective_correct / super_resolution / sharpen，偏视觉处理，缺业务工具（比价 / 防伪 / 退款 / 物流） |

同时，[answer/data.md](../answer/data.md) 已定调轨迹格式：**OpenAI Tool Call JSON**（结构化输出 + 并行调用 + 精确 Reward 信号），并明确「工具 Schema 定稿是数据构造的绝对前置依赖」。其结论直接作为本文档 5.2 节协议设计的依据：

1. 电商工具入参全是订单号、商品 ID、防伪码等精确字段，参数容错率极低，JSON 格式可从训练侧 + 推理侧双重保障参数正确；
2. 并行调用需求高频（一个订单号同时查物流和退款）；
3. GRPO 的奖励信号可精确到「工具名是否正确、参数是否合法、调用是否成功」；
4. OpenAI 格式是行业事实标准，Ollama / ms-swift / vLLM 原生支持。

### 2.2 目标

| 目标 | 衡量指标 |
|---|---|
| 工具收敛到 11 个（8-12 区间） | 23 工具收敛映射完成，无孤儿工具 |
| 工具调用格式合法率 ≥ 95% | 1000 题评测集（对齐 [01](01-product-overview.md) 5.1） |
| 参数溯源违规率 ≤ 2% | 评测集幻觉参数检测（编造 product_id / order_id） |
| 平均工具调用次数 ≤ 4 次/任务 | 成本控制（对齐 [01](01-product-overview.md) 5.1） |
| Schema 单一契约源 | 在线注册 / Teacher 生成 / SFT 数据 / GRPO 环境 4 处消费同一份 tools.json |
| 支持并行工具调用 | 多意图任务一次返回多个 tool_calls |

### 2.3 资源约束目标

- 11 个工具 Schema 注入 System Prompt 的 Token 预算 **≤ 2500 tokens**（超限压缩 description，不裁剪参数定义）。
- 单条工具 Observation（role=tool 消息）序列化后 **≤ 2 KB**（对齐 [03](03-business-flow.md) ToolResult 约束），超限截断并附 `truncated: true`。
- 单任务工具调用循环 ≤ 5 轮（MAX_TOOL_LOOP=5，[03](03-business-flow.md) 5.5.2），单轮并行调用 ≤ 3 条。
- 单条训练轨迹总长度 ≤ 8k tokens（含图片 Token），匹配云端 Teacher（Qwen3.8-27B Q4，33 GB VRAM）的上下文余量。
- 训练侧工具执行器（GRPO Rollout 环境）运行在云端训练机（域 C），不受本地三预算域限制。

---

## 3. 名词与缩写

| 缩写 | 含义 |
|---|---|
| Tool Schema | 工具的 JSON Schema 定义（名称、描述、参数、返回） |
| Tool Call | 模型发起的一次工具调用（name + arguments） |
| Observation | 工具执行结果，以 role=tool 消息回填对话 |
| Parallel Tool Call | 单次 assistant 输出携带多个 tool_calls |
| Constrained Decoding | 约束解码，强制输出符合 Schema 的合法 JSON |
| 参数溯源（Grounding） | 参数值必须来自先前 Observation、用户输入或记忆，禁止编造 |
| 参考工具链 | 某业务路由的最优工具调用序列，作为成本 Reward 基准 |
| Teacher | 轨迹生成的采样模型（云端 Qwen3.8-27B，[17](17-data-generation.md)） |
| image_ref | 会话内图片引用别名（img_1、img_2），避免在参数中传长 URL |
| MAX_TOOL_LOOP | 工具调用循环上限（5 轮，防死循环） |

---

## 4. 需求描述

### 4.1 功能性需求

| 编号 | 需求 |
|---|---|
| T-01 | 工具调用协议采用 OpenAI Tool Call JSON（tools 数组 + tool_calls 输出 + role=tool 回填） |
| T-02 | 工具体系收敛为 11 个：ocr、vl_describe、image_search、text_search、price_compare、authenticity_check、query_logistics、query_refund、create_refund_ticket、ask_user、transfer_to_human |
| T-03 | 每个工具有定稿 JSON Schema（参数名、类型、必填、枚举、描述），单一契约源 config/tools/tools.json |
| T-04 | 支持并行工具调用（无依赖的调用合并到同一轮 tool_calls） |
| T-05 | 参数溯源约束：product_id / order_id / 防伪码等必须来自合法来源，执行器校验 |
| T-06 | 工具分级：只读 8 个、写入 1 个（create_refund_ticket，过质量门控）、交互 2 个（ask_user / transfer_to_human） |
| T-07 | Observation 统一信封 {success, data / error}，≤ 2 KB 截断 |
| T-08 | 工具执行器双形态：在线形态（调真实服务）与训练形态（读 [17](17-data-generation.md) 数据资产），接口一致 |
| T-09 | 训练轨迹格式采用 ms-swift messages + tools 列，图片走 content 列表多模态格式 |
| T-10 | 每业务路由定义参考工具链与最少调用数，供成本 Reward 与数据筛选使用 |

### 4.2 非功能性需求

| 编号 | 维度 | 要求 |
|---|---|---|
| NT-01 | 质量 | 工具调用格式合法率 ≥ 95%；参数溯源违规率 ≤ 2% |
| NT-02 | 成本 | 平均调用次数 ≤ 4 次/任务；Schema Token ≤ 2500 |
| NT-03 | 时延 | 在线单工具执行超时 3s（[03](03-business-flow.md)），只读失败返回提示，写入失败转人工 |
| NT-04 | 可观测 | 每次调用落 cs_tool_call 审计表（tool_name、params、result、success、latency） |
| NT-05 | 可演进 | 新增工具只需追加 tools.json 与执行器实现，不改协议 |
| NT-06 | 一致性 | 训练环境与在线环境的 Observation 语义一致（同一数据资产） |

---

## 5. 详细设计

### 5.1 工具体系总览：23 工具收敛为 11 工具

#### 5.1.1 收敛原则

1. **确定性操作不进工具表**：图像预处理（裁剪 / 透视校正 / 超分 / 锐化 / 降噪 / 版面分析）是固定管线步骤（[03](03-business-flow.md) vision_preprocess_node），不需要模型决策，全部下沉为节点。
2. **同类合并**：object_detect 并入 vl_describe；product_search 并入 text_search（scope=product）；query_order 并入 query_logistics（订单状态随轨迹返回，[17](17-data-generation.md) 无独立订单表）。
3. **无数据支撑的删除**：web_search 删除——[05](05-rag-retrieval.md) 五级检索无 Web 级，业务数据全部来自本地数据资产（[17](17-data-generation.md)），引入外部搜索会增加依赖、时延与训练噪声。
4. **副作用异步化**：update_memory、save_knowledge_candidate 不是模型决策的动作，由 [03](03-business-flow.md) memory_write_node / knowledge_distill_node 异步管线触发。
5. **工具数量 8-12**：模型面对大工具表选择困惑（用户明确要求收敛），11 个在区间内且覆盖 6 大业务路由全部场景。

#### 5.1.2 收敛映射表

| 23 工具（[01](01-product-overview.md) 5.3） | 处置 | 去向 |
|---|---|---|
| ocr | **保留** | 工具 1：ocr |
| vl_describe | **保留** | 工具 2：vl_describe |
| image_search | **保留** | 工具 3：image_search |
| text_search | **保留** | 工具 4：text_search |
| price_compare | **保留** | 工具 5：price_compare |
| authenticity_check | **保留** | 工具 6：authenticity_check |
| query_logistics | **保留** | 工具 7：query_logistics（并入订单状态） |
| query_refund | **保留** | 工具 8：query_refund |
| create_refund_ticket | **保留** | 工具 9：create_refund_ticket（写入，过质量门控） |
| ask_user | **保留** | 工具 10：ask_user |
| transfer_to_human | **保留** | 工具 11：transfer_to_human |
| crop / perspective_correct / super_resolution / sharpen / denoise / layout_parsing（6 个） | 下沉 | vision_preprocess_node 固定管线（[03](03-business-flow.md) 5.4） |
| object_detect | 合并 | vl_describe（目标检测属视觉理解子任务） |
| product_search | 合并 | text_search（scope=product）/ image_search |
| web_search | 删除 | 五级 RAG 无 Web 级；本地数据资产闭环 |
| query_order | 合并 | query_logistics（返回 order_id 对应订单状态 + 轨迹） |
| update_memory | 下沉 | memory_write_node 异步管线（[03](03-business-flow.md) 5.8） |
| save_knowledge_candidate | 下沉 | knowledge_distill_node 异步管线（[03](03-business-flow.md) 5.8） |

> [01](01-product-overview.md) 5.3 的「23 工具为目标、落地优先实现核心子集」以本表为准；[03](03-business-flow.md) 中「23 工具受控调用」的表述同步收敛为 11 工具。

#### 5.1.3 工具分类与权限

| 类别 | 工具 | 权限 | 说明 |
|---|---|---|---|
| 视觉理解（2） | ocr、vl_describe | 只读 | 图片到文本 / 结构化判定 |
| 检索（2） | image_search、text_search | 只读 | 向量召回（[05](05-rag-retrieval.md) L2）与知识检索（L1/L3） |
| 业务查询（4） | price_compare、authenticity_check、query_logistics、query_refund | 只读 | 结构化精确查询（[05](05-rag-retrieval.md) L4） |
| 写入（1） | create_refund_ticket | 写入（高风险） | 需瑕疵证据 + 质量门控（[08](08-quality-and-human.md)） |
| 交互（2） | ask_user、transfer_to_human | 特殊 | 观察者是用户 / 人工坐席 |

### 5.2 工具调用协议（OpenAI Tool Call JSON）

#### 5.2.1 协议流程

```text
1. 注入：System Prompt（角色 + 行为规则）+ tools 数组（11 个 JSON Schema）
   ↓
2. 用户消息：文本 + 图片（qwen3.5:4b 原生多模态，content 列表）
   ↓
3. 模型输出 assistant 消息，二选一：
   a. tool_calls 数组（1..3 个并行调用，每个含 id / name / arguments-JSON字符串）
   b. 纯 content 文本回答（任务完成，循环终止）
   ↓
4. 执行器逐条执行，回填 role=tool 消息（Observation 信封，见 5.4）
   ↓
5. 回到步骤 3，直到输出纯文本 或 达 MAX_TOOL_LOOP=5 强制生成
```

#### 5.2.2 关键约定

| 约定 | 内容 |
|---|---|
| 格式标准 | OpenAI Chat Completions tool calling 格式（行业事实标准，Ollama `/api/chat` 的 `tools` 参数、ms-swift、vLLM 均原生支持） |
| 基座能力 | qwen3.5:4b 原生多模态 + 原生 tool call（训练与在线同模型，微调产物直接回灌 Ollama，见 [02](02-architecture.md)） |
| thinking 模式 | 在线工具循环与训练轨迹**均关闭 thinking**（Ollama `think=false`；ms-swift 模板对应开关），轨迹不含思考段；GRPO 阶段是否开启 thinking 待超参调试阶段评估 |
| image_ref 别名 | 会话内图片统一以 `img_1`、`img_2` 引用（对应 [03](03-business-flow.md) image_metas 下标），执行器解析为 URL。避免模型抄写长 URL 出错，参数溯源友好 |
| 并行调用 | 无依赖的调用合并到同一轮 tool_calls（如同一订单查物流 + 退款）；有依赖必须分轮（image_search → price_compare） |
| 循环上限 | MAX_TOOL_LOOP=5 轮（一轮 = 一次 assistant tool_calls，不论并行条数）；调用次数（成本计数）按工具条数累计 |
| 约束解码 | 推理侧可开启约束解码（Ollama `format=json` / vLLM guided decoding）强制输出合法 JSON，工业标配兜底（[answer/data.md](../answer/data.md) 补充建议） |
| 轨迹终止 | 模型输出不含 tool_calls 的 content 即终止；ask_user 的 Observation 为用户下一条消息（继续循环） |

#### 5.2.3 协议消息骨架

```json
{
  "messages": [
    {"role": "system", "content": "你是多模态电商客服 Agent...（行为规则，见 5.11）"},
    {"role": "user", "content": [
      {"type": "image", "image": "img_1 实际 URL 或训练数据相对路径"},
      {"type": "text", "text": "帮我看看这个同款多少钱"}
    ]},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_001", "type": "function",
       "function": {"name": "image_search", "arguments": "{\"image_ref\": \"img_1\", \"top_k\": 3}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_001", "name": "image_search",
     "content": "{\"success\": true, \"data\": {\"candidates\": [...]}}"}
  ],
  "tools": ["config/tools/tools.json 中的 11 个 Schema"]
}
```

### 5.3 工具 Schema 定稿（11 个）

> 以下为 `config/tools/tools.json` 的权威定义。description 已按 Token 预算精炼，**修改 Schema 必须同步评估对已生成训练数据的影响**（Schema 变更 = 数据版本变更）。

#### 5.3.1 ocr（图片文字提取）

| 属性 | 值 |
|---|---|
| 用途 | 从图片提取文字：订单截图的订单号 / 金额 / 状态、防伪码标签、小票 |
| 权限 | 只读 |
| 使用路由 | logistics_route、authenticity_route、refund_route、refund_track_route |
| 前置约束 | image_ref 必须是当前会话图片（img_N） |

```json
{
  "type": "function",
  "function": {
    "name": "ocr",
    "description": "对图片做文字识别(OCR)。用于从订单截图、防伪码标签、小票中提取订单号、金额、防伪码等文字。图片需已通过质量门控(清晰可读)。",
    "parameters": {
      "type": "object",
      "properties": {
        "image_ref": {
          "type": "string",
          "description": "会话内图片引用，如 img_1、img_2"
        },
        "focus": {
          "type": "string",
          "enum": ["order_id", "authenticity_code", "amount", "all"],
          "description": "关注的字段类型，默认 all 全量提取"
        }
      },
      "required": ["image_ref"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {"blocks": [
  {"text": "订单号：ORD00001042", "confidence": 0.98},
  {"text": "金额：￥399.00", "confidence": 0.96},
  {"text": "状态：运输中", "confidence": 0.94}
]}}
```

失败模式：图片无文字（`error.code=NO_TEXT`）、图片模糊不可识别（`BLURRY`）、OCR 服务超时（`TIMEOUT`，降级路径见 [03](03-business-flow.md) 视觉预处理）。

#### 5.3.2 vl_describe（视觉理解分析）

| 属性 | 值 |
|---|---|
| 用途 | 视觉理解与判定：瑕疵类型与位置、商品类目 / 品牌 / 属性、包装细节、是否商品图 |
| 权限 | 只读 |
| 使用路由 | refund_route（瑕疵判定）、same_item_route（属性辅助）、image_quality 兜底 |
| 前置约束 | image_ref 必须是当前会话图片；question 需明确分析目标 |

```json
{
  "type": "function",
  "function": {
    "name": "vl_describe",
    "description": "视觉理解分析。输入图片与分析指令，返回结构化判定结论。用于瑕疵判定(类型/位置/严重程度)、商品属性提取(类目/品牌/颜色)、包装外观检查、判断图片是否为商品图。",
    "parameters": {
      "type": "object",
      "properties": {
        "image_ref": {"type": "string", "description": "会话内图片引用，如 img_1"},
        "question": {
          "type": "string",
          "description": "分析指令，如：判断该商品是否存在瑕疵，输出瑕疵类型与位置"
        }
      },
      "required": ["image_ref", "question"]
    }
  }
}
```

返回示例（瑕疵判定）：

```json
{"success": true, "data": {
  "conclusion": "存在瑕疵",
  "details": "鞋面右侧有明显开胶，位于鞋头与鞋身接缝处，长约 3cm",
  "tags": ["defect", "开胶", "鞋面"]
}}
```

失败模式：图片非商品图（`NOT_PRODUCT`）、判定置信度低（`LOW_CONFIDENCE`，建议 ask_user 补图）。

#### 5.3.3 image_search（以图搜图）

| 属性 | 值 |
|---|---|
| 用途 | 以图搜图，同款 / 相似商品召回（Top-K） |
| 权限 | 只读 |
| 使用路由 | same_item_route、consult_route（图搜辅助） |
| 前置约束 | image_ref 必须是当前会话图片；返回的 product_id 供 price_compare / text_search 使用（参数溯源） |
| 实现依赖 | [05](05-rag-retrieval.md) L2：bge-vl-large 图片向量 + pgvector HNSW，商品图特征预计算 |

```json
{
  "type": "function",
  "function": {
    "name": "image_search",
    "description": "以图搜图，在商品库中查找同款或相似商品，返回 Top-K 候选(含 product_id、标题、类目、价格、相似度)。用于同款识别与比价入口。拿到候选后可用 product_id 调 price_compare。",
    "parameters": {
      "type": "object",
      "properties": {
        "image_ref": {"type": "string", "description": "会话内图片引用，如 img_1"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "返回候选数，默认 5"},
        "category": {"type": "string", "description": "类目过滤(3c/clothing/home/food)，可选"}
      },
      "required": ["image_ref"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {"candidates": [
  {"product_id": 1042, "title": "VogueStep VS-2203 轻量跑步鞋 白色 42码", "category": "clothing", "price": 399.0, "similarity": 0.92},
  {"product_id": 2317, "title": "VogueStep VS-2201 轻量跑步鞋 黑色 41码", "category": "clothing", "price": 379.0, "similarity": 0.85}
]}}
```

失败模式：候选为空（`NO_MATCH`，建议 ask_user 补拍角度）、向量服务不可用（降级文本检索，[05](05-rag-retrieval.md) 7.2）。

#### 5.3.4 text_search（知识检索）

| 属性 | 值 |
|---|---|
| 用途 | 知识检索统一入口：FAQ、售后政策文档、商品信息 |
| 权限 | 只读 |
| 使用路由 | consult_route、after_sale_policy、refund_route（政策查询） |
| 实现依赖 | [05](05-rag-retrieval.md) L1（FAQ）+ L3（文档）+ L2（scope=product） |

```json
{
  "type": "function",
  "function": {
    "name": "text_search",
    "description": "知识库文本检索(FAQ/售后政策/商品信息)。返回相关文档片段。用于商品咨询、退换货政策、保修条款等问题；scope=product 时按关键词查商品信息。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "检索查询，如：七天无理由退货政策"},
        "scope": {"type": "string", "enum": ["faq", "policy", "product", "all"], "description": "检索范围，默认 all"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "返回条数，默认 5"}
      },
      "required": ["query"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {"docs": [
  {"source_type": "faq", "title": "七天无理由退货", "content": "自签收之日起 7 天内，商品未使用且不影响二次销售的，可申请无理由退货...", "score": 0.91}
]}}
```

失败模式：无命中（`NO_MATCH`）、向量库失败降级 BM25（[05](05-rag-retrieval.md) 7.2）。

#### 5.3.5 price_compare（跨平台比价）

| 属性 | 值 |
|---|---|
| 用途 | 跨平台比价（jd / taobao / pdd / amazon），返回各平台价格与最低价 |
| 权限 | 只读 |
| 使用路由 | same_item_route |
| 前置约束 | **product_id 必须来自 image_search 或 text_search 返回的候选**，禁止编造（执行器校验，见 5.5） |
| 实现依赖 | [17](17-data-generation.md) cs_product_price 表（5000 SKU × 4 平台） |

```json
{
  "type": "function",
  "function": {
    "name": "price_compare",
    "description": "跨平台比价。输入商品ID(必须来自 image_search 或 text_search 的返回结果)，返回京东/淘宝/拼多多/亚马逊四平台价格与最低价。用于同款比价场景。",
    "parameters": {
      "type": "object",
      "properties": {
        "product_id": {"type": "integer", "description": "商品ID，必须来自检索工具返回的候选"},
        "platforms": {
          "type": "array",
          "items": {"type": "string", "enum": ["jd", "taobao", "pdd", "amazon"]},
          "description": "指定平台，默认全部四平台"
        }
      },
      "required": ["product_id"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {
  "product_id": 1042,
  "prices": [
    {"platform": "jd", "price": 399.0},
    {"platform": "taobao", "price": 369.0},
    {"platform": "pdd", "price": 349.5},
    {"platform": "amazon", "price": 419.0}
  ],
  "lowest": {"platform": "pdd", "price": 349.5}
}}
```

失败模式：product_id 不存在（`NOT_FOUND`）、product_id 溯源违规（`GROUNDING_VIOLATION`，参数不在先前 Observation 中）。

#### 5.3.6 authenticity_check（防伪验证）

| 属性 | 值 |
|---|---|
| 用途 | 防伪码验证，返回真伪结论 |
| 权限 | 只读 |
| 使用路由 | authenticity_route |
| 前置约束 | code 必须来自 ocr 返回或用户文本，禁止编造 |
| 实现依赖 | [17](17-data-generation.md) cs_anti_fake_code 表（5000 码，5% 假） |

```json
{
  "type": "function",
  "function": {
    "name": "authenticity_check",
    "description": "防伪码验证。输入防伪码(来自图片OCR识别或用户提供的文本)，查询防伪库返回真伪结论与验证次数。防伪码通常印在商品包装或标签上。",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "防伪码，如 AF00001042K，来自OCR或用户输入"}
      },
      "required": ["code"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {"code": "AF00001042K", "is_genuine": true, "product_title": "VogueStep VS-2203 轻量跑步鞋", "verify_count": 1}}
```

失败模式：防伪码未登记（`NOT_REGISTERED`，提示用户核对或转人工）、格式非法（`INVALID_FORMAT`）。

#### 5.3.7 query_logistics（物流轨迹查询）

| 属性 | 值 |
|---|---|
| 用途 | 订单物流轨迹查询（含订单当前状态），兼容订单状态查询意图 |
| 权限 | 只读 |
| 使用路由 | logistics_route、order_query 意图 |
| 前置约束 | order_id 必须来自 ocr 返回、用户文本或记忆，禁止编造 |
| 实现依赖 | [17](17-data-generation.md) cs_logistics_track 表（5000 单 × 4-6 轨迹点，5 种状态） |

```json
{
  "type": "function",
  "function": {
    "name": "query_logistics",
    "description": "查询订单物流轨迹。输入订单号(来自订单截图OCR、用户输入或历史记录)，返回订单当前状态与完整物流轨迹(已发货/运输中/派送中/已签收/已拒收)。也可用于查询订单当前状态。",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {"type": "string", "description": "订单号，如 ORD00001042"}
      },
      "required": ["order_id"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {
  "order_id": "ORD00001042",
  "status": "in_transit",
  "trajectory": [
    {"ts": "2026-08-12 10:30:00", "location": "杭州", "action": "shipped"},
    {"ts": "2026-08-13 08:30:00", "location": "南京", "action": "in_transit"}
  ]
}}
```

失败模式：订单号不存在（`NOT_FOUND`，提示用户核对）、格式非法（`INVALID_FORMAT`）。

#### 5.3.8 query_refund（退款进度查询）

| 属性 | 值 |
|---|---|
| 用途 | 退款进度查询（按订单号或退款单号） |
| 权限 | 只读 |
| 使用路由 | refund_track_route（进度查询，不建单） |
| 前置约束 | order_id / refund_id 必须来自 ocr、用户文本或记忆，禁止编造 |
| 实现依赖 | [17](17-data-generation.md) cs_refund_ticket 表（500 单，覆盖 [09](09-ticket-generation.md) 5 状态） |

```json
{
  "type": "function",
  "function": {
    "name": "query_refund",
    "description": "查询退款进度。输入订单号或退款单号，返回退款工单当前状态(init/reviewing/approved/refunded/rejected)、退款原因、金额与更新时间。用于用户查询退款处理到哪一步了。",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {"type": "string", "description": "订单号，与 refund_id 二选一"},
        "refund_id": {"type": "string", "description": "退款单号，如 RF00000001，与 order_id 二选一"}
      },
      "required": []
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {
  "refund_id": "RF00000001", "order_id": "ORD00001042",
  "state": "reviewing", "reason": "客服审核中，需补充瑕疵图",
  "amount": 399.0, "updated_at": "2026-08-18 14:00:00"
}}
```

失败模式：无对应退款单（`NOT_FOUND`）、参数二选一缺失（`MISSING_PARAM`）。

#### 5.3.9 create_refund_ticket（创建退款工单，写入类）

| 属性 | 值 |
|---|---|
| 用途 | 创建退款工单（唯一写入类工具） |
| 权限 | 写入，高风险 |
| 使用路由 | refund_route（建单） |
| 前置约束 | **调用前必须有瑕疵证据**（vl_describe / ocr 的 Observation）或用户明确陈述；order_id 必须来自合法来源 |
| 风控 | [03](03-business-flow.md) quality_gate_node：质量分 < 0.70 且高风险场景直接转人工，不建单；[09](09-ticket-generation.md) 退款状态机 init 起步 |

```json
{
  "type": "function",
  "function": {
    "name": "create_refund_ticket",
    "description": "创建退款工单。需先有瑕疵证据(vl_describe/ocr结果)或用户明确陈述的退款理由。创建成功返回退款单号，初始状态 init 待审核。仅在证据充分时调用；证据不足或用户意图不明时优先 ask_user。",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {"type": "string", "description": "订单号，来自OCR/用户输入/退款查询"},
        "reason": {
          "type": "string",
          "enum": ["quality_flaw", "wrong_item", "damaged_in_shipping", "unwanted", "other"],
          "description": "退款原因：质量瑕疵/错发/运输破损/不想要了/其他"
        },
        "flaw_description": {"type": "string", "description": "瑕疵描述，来自 vl_describe 结果，如：鞋面右侧开胶约3cm"},
        "amount": {"type": "number", "description": "退款金额，来自订单信息，可选"}
      },
      "required": ["order_id", "reason"]
    }
  }
}
```

返回示例：

```json
{"success": true, "data": {"refund_id": "RF00000501", "state": "init", "created": true}}
```

失败模式：order_id 不存在（`NOT_FOUND`）、重复建单（`DUPLICATED`，同一订单已有进行中退款单）、写入服务失败（转人工，[03](03-business-flow.md) 写入类失败策略）。

#### 5.3.10 ask_user（追问用户）

| 属性 | 值 |
|---|---|
| 用途 | 追问澄清：缺槽位（订单号）、缺图 / 补图（模糊、非商品图）、候选确认（多个同款选哪个） |
| 权限 | 特殊（Observation 为用户下一条消息） |
| 使用路由 | 全路由（槽位不全 / need_image / 多候选） |
| 前置约束 | 单任务追问 ≤ 2 次（[03](03-business-flow.md) 单轮主动追问最多 1 轮 + 质量门控补图 1 次），超过直接转人工 |

```json
{
  "type": "function",
  "function": {
    "name": "ask_user",
    "description": "向用户追问。用于：缺少订单号等必要信息、图片模糊/非商品图需补拍、同款候选多个需用户确认。question 要具体明确，一次只问一件事。",
    "parameters": {
      "type": "object",
      "properties": {
        "question": {"type": "string", "description": "追问内容，如：麻烦提供一下订单号，或发送订单截图"},
        "expect": {
          "type": "string",
          "enum": ["text", "image", "order_id", "choice"],
          "description": "期望回复类型，可选"
        }
      },
      "required": ["question"]
    }
  }
}
```

返回：Observation 即用户下一条消息（文本或图片），由执行器包装为 `{"success": true, "data": {"user_reply": "..."}}`。

#### 5.3.11 transfer_to_human（转人工）

| 属性 | 值 |
|---|---|
| 用途 | 转人工坐席，传递上下文摘要 |
| 权限 | 特殊（终止工具循环） |
| 使用路由 | 高风险低置信（退款 / 防伪争议）、写入失败、用户明确要求、投诉 |
| 前置约束 | 高风险意图置信度 < 0.70 直接转人工（[04](04-intent-routing.md) 5.9.3），无需模型决策 |

```json
{
  "type": "function",
  "function": {
    "name": "transfer_to_human",
    "description": "转接人工客服。用于：退款/防伪争议、用户情绪激动或投诉、工具连续失败、用户明确要求人工。调用后附带 reason 与 summary 供坐席快速接管。",
    "parameters": {
      "type": "object",
      "properties": {
        "reason": {"type": "string", "description": "转人工原因，如：退款争议需人工核实"},
        "summary": {"type": "string", "description": "会话摘要：用户问题、已做的事、当前卡点"}
      },
      "required": ["reason"]
    }
  }
}
```

返回：`{"success": true, "data": {"handoff": true, "ticket": "..."}}`，工具循环终止。

### 5.4 Observation 统一信封

所有工具（ask_user / transfer_to_human 除外）的返回统一为：

```json
{"success": true, "data": {"...": "..."}}

{"success": false, "error": {"code": "NOT_FOUND", "message": "订单号不存在，请核对后重试"}}
```

| 规则 | 内容 |
|---|---|
| 体积 | 序列化 ≤ 2 KB；超限截断 data 并附 `"truncated": true`，明细落 cs_tool_call 表 |
| 错误码 | 通用：NOT_FOUND / INVALID_FORMAT / MISSING_PARAM / TIMEOUT / GROUNDING_VIOLATION / NO_MATCH / BLURRY / NOT_PRODUCT / NO_TEXT / LOW_CONFIDENCE / NOT_REGISTERED / DUPLICATED / INTERNAL |
| 错误可恢复性 | 模型收到 error 后可修正参数重试（计调用次数）或改走 ask_user / transfer_to_human |
| 审计 | 每次调用写 cs_tool_call（[03](03-business-flow.md) 6.2 事件 cs.tool.call.audit） |

### 5.5 参数溯源约束（Grounding，Reward 依据）

| 参数 | 合法来源 | 违规判定 |
|---|---|---|
| image_ref | 当前会话图片（img_N） | 引用不存在的图片序号 |
| price_compare.product_id | image_search / text_search 返回的 candidates.product_id | 编造未出现过的 product_id |
| authenticity_check.code | ocr 返回的 blocks.text / 用户文本 | 编造防伪码 |
| query_logistics.order_id | ocr 返回 / 用户文本 / 记忆快照 | 编造订单号 |
| query_refund.order_id / refund_id | 同上 / query_refund 上下文 | 编造单号 |
| create_refund_ticket.order_id | ocr / 用户文本 / query_refund 返回 | 编造订单号 |
| create_refund_ticket.flaw_description | vl_describe / ocr 的 Observation | 无证据建单（幻觉瑕疵） |

执行器（在线与训练两形态）统一校验：违规直接返回 `GROUNDING_VIOLATION` 错误，不执行查询。该约束同时是：

1. **SFT 数据筛选项**：Teacher 轨迹中含违规调用的样本剔除或修正；
2. **GRPO 过程质量 Reward** 的扣分项（[01](01-product-overview.md) 5.4.4 w3）。

### 5.6 各业务路由参考工具链（成本 Reward 基准）

| 路由 | 意图（[04](04-intent-routing.md)） | 参考工具链 | 最少调用数 n_ref |
|---|---|---|---|
| consult_route | product_consult / after_sale_policy | text_search | 1 |
| same_item_route | same_item_price_compare | image_search → price_compare | 2 |
| authenticity_route | authenticity_check | ocr → authenticity_check | 2 |
| logistics_route | logistics_query / order_query | ocr → query_logistics | 2 |
| refund_route（瑕疵图） | refund_request | vl_describe → create_refund_ticket | 2 |
| refund_route（纯文本+订单号） | refund_request | create_refund_ticket | 1 |
| refund_track_route | 退款进度 | query_refund | 1 |
| 多跳复合（图 + 双意图） | 同款比价 + 退款 | image_search → price_compare ∥ query_refund | 3 |
| 缺图 / 模糊 | 任意图图前置 | ask_user | 1 |
| 高风险争议 | refund / authenticity 低置信 | transfer_to_human | 1 |

> 并行符号 ∥ 表示两调用无依赖可合并一轮。**成本 Reward（[01](01-product-overview.md) 5.4.4 w4=0.1）**：实际调用条数 ≤ n_ref 得满分，每超出 1 条按比例扣分；参考链同时用于 SFT 数据的「最优轨迹」模板与 GRPO 高难度题的构造约束（多跳题按 3-5 条链长构造）。

### 5.7 并行工具调用设计

| 场景 | 并行调用 |
|---|---|
| 订单截图问「物流到哪了，退款到账没」 | ocr（单条）→ [query_logistics ∥ query_refund]（同 order_id，无依赖） |
| 商品图问「同款多少钱，七天能退吗」 | image_search → [price_compare ∥ text_search]（policy 检索无依赖） |
| 禁止并行 | 后一调用参数依赖前一 Observation（image_search → price_compare 必须分轮） |

训练数据中并行轨迹占比约 15%（多意图场景），保证模型学会合并无依赖调用（[answer/data.md](../answer/data.md) 并行调用优势）。单轮并行上限 3 条。

### 5.8 写入类工具风控（create_refund_ticket）

```text
模型发起 create_refund_ticket
  |
  v
执行器前置校验：
  ├── order_id 溯源合法？
  ├── 已有 vl_describe/ocr 证据 或 用户明确陈述？
  └── 同订单无进行中退款单（幂等）？
  |
  v
quality_gate_node（[03] 5.6 / [08]）：
  ├── 质量分 ≥ 0.85 → 建单，返回 refund_id
  ├── 0.70 ~ 0.85 → 建单，回复标注「待人工复核」
  ├── < 0.70 且高风险 → 拦截建单，转 human_handoff
  └── 缺图证据 → need_image 分支，ask_user 补瑕疵图，不建单
```

> 训练环境（GRPO Rollout）同样实现该前置校验，保证「错误建单」在训练轨迹中得到负奖励（最终答案 Reward 扣分）。

### 5.9 训练轨迹数据格式（ms-swift）

SFT 数据采用 ms-swift messages 格式（与 [02](02-architecture.md) 训练框架选型一致），JSONL 每行一条轨迹：

```json
{
  "messages": [
    {"role": "system", "content": "你是多模态电商客服 Agent。根据用户输入(含图片)选择并调用工具；参数必须来自工具返回或用户输入，禁止编造；任务完成后用中文简洁回答。当前会话图片：img_1。"},
    {"role": "user", "content": [
      {"type": "image", "image": "data/images/products/sneaker_0042.jpg"},
      {"type": "text", "text": "帮我看看这个同款多少钱，哪个平台最便宜"}
    ]},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_001", "type": "function",
       "function": {"name": "image_search", "arguments": "{\"image_ref\": \"img_1\", \"top_k\": 3}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_001", "name": "image_search",
     "content": "{\"success\": true, \"data\": {\"candidates\": [{\"product_id\": 1042, \"title\": \"VogueStep VS-2203 轻量跑步鞋 白色 42码\", \"category\": \"clothing\", \"price\": 399.0, \"similarity\": 0.92}]}}"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_002", "type": "function",
       "function": {"name": "price_compare", "arguments": "{\"product_id\": 1042}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_002", "name": "price_compare",
     "content": "{\"success\": true, \"data\": {\"product_id\": 1042, \"prices\": [{\"platform\": \"jd\", \"price\": 399.0}, {\"platform\": \"taobao\", \"price\": 369.0}, {\"platform\": \"pdd\", \"price\": 349.5}, {\"platform\": \"amazon\", \"price\": 419.0}], \"lowest\": {\"platform\": \"pdd\", \"price\": 349.5}}}"},
    {"role": "assistant", "content": "为您找到同款：VogueStep VS-2203 轻量跑步鞋。各平台价格：拼多多最低 ¥349.5，淘宝 ¥369.0，京东 ¥399.0，亚马逊 ¥419.0，建议在拼多多购买最划算。"}
  ],
  "tools": ["config/tools/tools.json 中的 11 个 Schema 原样嵌入"]
}
```

| 约定 | 内容 |
|---|---|
| 损失掩码 | 仅 assistant 轮计损失（含 tool_calls 与最终 content），user / tool 轮不计（ms-swift 模板自动处理） |
| 图片 | 仅出现在 user 消息 content 列表；训练图片为 [17](17-data-generation.md) 采集的真实商品图（Unsplash） |
| tools 字段 | 每条轨迹嵌入同一份 11 工具 Schema（单一契约源拷贝），保证与在线注入一致 |
| 轨迹长度 | ≤ 8k tokens；图片 Token 计入（qwen3.5:4b 图像编码） |
| 多样性 | 6 大路由全覆盖 + 多跳复合 + 缺图追问 + 高风险转人工 + 并行调用（~15%） |
| 筛选 | Teacher 采样多条 → 执行器真实执行 → 保留成功且溯源合法的轨迹（[01](01-product-overview.md) 5.4.2） |

### 5.10 工具执行器双形态

| 形态 | 运行位置 | 数据源 | 用途 |
|---|---|---|---|
| 在线形态 | 本地 Docker（域 A） | PostgreSQL + pgvector + MinIO + bge-m3 + 视觉预处理服务 | LangGraph tool_call_node 真实调用 |
| 训练形态 | 云端训练机（域 C） | [17](17-data-generation.md) 生成的 JSONL 数据资产（products / prices / anti_fake / logistics / refunds，内存索引，总量 < 50 MB） | Teacher 轨迹生成与 GRPO Rollout 的真实 Observation |

两种形态实现同一 `ToolExecutor` 接口（11 个工具 + 溯源校验 + Observation 信封），保证训练与在线的 Observation 语义一致（「环境执行工具得到真实 Observation，不靠模拟」，[01](01-product-overview.md) 5.4.3）。

> 训练形态将 JSONL 全量载入训练机内存属域 C 资源（云端训练机内存充裕），不受本地「禁止全量加载」约束（该约束针对域 A 在线服务）。

### 5.11 System Prompt 行为规则（注入模板）

```text
你是多模态电商客服 Agent。根据用户输入(文本+图片)选择并调用工具解决问题。
规则：
1. 参数必须来自工具返回结果或用户输入，禁止编造订单号、商品ID、防伪码。
2. 无依赖的多个查询合并到同一轮并行调用；有依赖的必须等前序结果。
3. 图片模糊、非商品图或缺关键信息时，用 ask_user 追问，不要猜测。
4. 退款建单前必须有瑕疵证据(vl_describe/ocr)或用户明确陈述；证据不足先追问。
5. 退款/防伪争议、用户投诉、连续工具失败时，调用 transfer_to_human 并附摘要。
6. 任务完成后直接用中文简洁回答(200字内)，不要再调用工具。
当前会话图片：{image_refs列表}。
```

### 5.12 Token 预算

| 项 | 预算 | 说明 |
|---|---|---|
| tools Schema 注入 | ≤ 2500 tokens | 11 个工具约 2000-2300，description 已精炼；新增工具需重估 |
| System Prompt | ≤ 300 tokens | 5.11 模板 |
| 单条 Observation | ≤ 2 KB（约 600 tokens） | 截断策略见 5.4 |
| 单轨迹总长 | ≤ 8k tokens | 匹配 Teacher（27B Q4）上下文余量与训练序列长度 |
| 图片 | 单图编码 ≤ 1500 tokens | 分辨率限制（[03](03-business-flow.md) ≤ 2048px 预处理） |

---

## 6. 数据与接口

### 6.1 工具契约单一源

```text
config/
└── tools/
    ├── tools.json          # 11 个工具 OpenAI Schema（权威定义，本 PRD 5.3 的机器可读版）
    └── executor.py         # ToolExecutor 接口（在线/训练双形态共用签名）
```

消费方（只读）：在线 LangGraph 注册、Teacher 轨迹生成 Prompt、SFT 数据集 tools 字段、GRPO Rollout 环境。

### 6.2 在线服务接口（Python AI 服务内部）

| 接口 | 方法 | 用途 |
|---|---|---|
| `POST /ai/tools/execute` | POST | 工具执行统一入口（tool_call_node 调用，含溯源校验） |
| `GET /ai/tools` | GET | 返回 tools.json（调试 / 前端展示） |
| `POST /ai/tools/validate` | POST | 校验一批 tool_calls 的格式与溯源（测试用） |

### 6.3 训练数据契约（对接 [01](01-product-overview.md) 5.4 与 [17](17-data-generation.md)）

| 文件 | 格式 | 规模（20k 档） | 说明 |
|---|---|---|---|
| data/sft/public_qa.jsonl | messages（5k 无 tools / 3k 带 tools） | 8000 | 公开多模态 QA：5k 纯 VQA（视觉底座，防工具格式过拟合）+ 3k 工具化改造（合成 ocr/vl_describe 轨迹），详见 [18](18-training-data.md) 5.2.1 |
| data/sft/ecommerce.jsonl | messages + tools | 6000 | 电商构造 QA（真实商品元数据构造） |
| data/sft/trajectories.jsonl | messages + tools | 6000 | Teacher（Qwen3.8-27B）采样轨迹，执行器真实执行筛选 |
| data/rl/prompts.jsonl | question + gold + n_ref | 8000 | GRPO 题集（金标答案 + 参考链长度，Reward 判分依据） |

> 规模扩至 50k 档时按同比例放大 trajectories 部分（公开与电商构造相对固定）。

---

## 7. 边界与异常

### 7.1 边界

- 本期工具固定 11 个，不扩表；新增工具需评估 Schema Token 预算与训练数据兼容性（Schema 变更 = 数据版本变更）。
- 不做多图联合推理（多图按 image_ref 顺序独立处理，[03](03-business-flow.md) 7.1）。
- 图像预处理（裁剪 / 透视校正 / 超分 / 锐化 / 降噪 / 版面分析）不是工具，是 vision_preprocess_node 固定管线。
- 本期不支持工具流式结果回传（Observation 一次性返回）。
- GRPO 阶段 thinking 模式默认关闭，开启与否属超参调试范畴。

### 7.2 异常处理

| 异常 | 处理 |
|---|---|
| 模型输出非法 JSON / 未知工具名 | 约束解码拦截；未拦截时返回 `INVALID_FORMAT` Observation 供模型自纠，重试计入调用次数 |
| 参数溯源违规 | 返回 GROUNDING_VIOLATION，不执行；GRPO 中计入过程扣分 |
| 只读工具超时（> 3s） | 返回 TIMEOUT 提示，不阻塞；连续 2 次失败建议转人工 |
| 写入工具失败 | 转人工（[03](03-business-flow.md) 写入类失败策略），不自动重试 |
| Observation 超 2 KB | 截断 + truncated 标记，明细落 cs_tool_call |
| 并行调用部分失败 | 成功的正常回填，失败的回填 error；模型可单独重试失败项 |
| ask_user 后用户仍缺信息 | 第 2 次追问仍不全则转人工（追问 ≤ 2 次） |
| 工具循环达 MAX_TOOL_LOOP=5 | 强制进入生成节点，用已有信息回答并标注不确定（[03](03-business-flow.md) 5.5.2） |
| 训练环境数据资产缺失 | Rollout 前置校验 5 份 JSONL 行数，缺失则报错终止（不静默降级） |

### 7.3 资源约束（强制）

- tools Schema ≤ 2500 tokens；单 Observation ≤ 2 KB；单轨迹 ≤ 8k tokens。
- MAX_TOOL_LOOP=5；单轮并行 ≤ 3 条；单任务追问 ≤ 2 次。
- 训练形态执行器运行在云端（域 C），全量 JSONL 内存索引 < 50 MB，不占本地预算。
- 在线形态走 Docker（域 A），工具执行读写 PostgreSQL / MinIO / bge-m3，禁止在 Python AI 镜像引入 torch。
- 详细约束见 [overview.md](overview.md) 第八节与 [../CLAUDE.md](../CLAUDE.md) 第五节。

---

## 8. 验收标准

- [ ] config/tools/tools.json 定稿，11 个 Schema 与本 PRD 5.3 完全一致，机器可校验（JSON Schema 校验通过）。
- [ ] 11 个工具在线形态全部实现并通过单元测试（含溯源校验、Observation 信封、截断）。
- [ ] 训练形态执行器（JSONL 内存索引）与在线形态对同一输入产出语义一致的 Observation（抽样比对）。
- [ ] 工具调用格式合法率 ≥ 95%（1000 题评测集，约束解码关闭下测模型原生能力）。
- [ ] 参数溯源违规率 ≤ 2%（幻觉 product_id / order_id 检测）。
- [ ] 平均工具调用次数 ≤ 4 次/任务；各路由实际调用分布与参考链（5.6）偏差 ≤ 1。
- [ ] 并行调用在多意图场景正确合并（评测集并行场景 ≥ 80% 合并率）。
- [ ] create_refund_ticket 过质量门控：无证据建单被拦截，拦截率 100%。
- [ ] tools Schema 注入 Token ≤ 2500（tokenizer 实测）。
- [ ] SFT 样例轨迹（每路由 ≥ 1 条）通过 ms-swift 数据校验（swift pt --dataset 校验）。
- [ ] 每次工具调用落 cs_tool_call 审计表。

---

## 9. 关联文档

- [01-product-overview.md](01-product-overview.md)：23 工具体系概览与训练闭环（本文档收敛其 5.3 节）。
- [02-architecture.md](02-architecture.md)：训练框架 ms-swift、qwen3.5:4b 基座。
- [03-business-flow.md](03-business-flow.md)：tool_call_node、vision_preprocess_node（预处理下沉依据）、MAX_TOOL_LOOP。
- [04-intent-routing.md](04-intent-routing.md)：意图与路由（参考工具链的意图侧输入）。
- [05-rag-retrieval.md](05-rag-retrieval.md)：image_search / text_search 的检索实现（L1-L4）。
- [07-model-fallback.md](07-model-fallback.md)：工具调用模型的降级链。
- [08-quality-and-human.md](08-quality-and-human.md)：写入类工具质量门控。
- [09-ticket-generation.md](09-ticket-generation.md)：退款状态机（create_refund_ticket / query_refund 数据语义）。
- [17-data-generation.md](17-data-generation.md)：工具执行器（训练形态）依赖的数据资产。
- [answer/data.md](../answer/data.md)：轨迹格式选型依据（OpenAI Tool Call JSON）。
- [overview.md](overview.md)：第八节内存占用分析。
- [../CLAUDE.md](../CLAUDE.md)：资源约束与禁止事项。

---

## 10. 变更记录

| 版本 | 日期 | 变更人 | 变更说明 |
|---|---|---|---|
| v1.0 | 2026-08-19 | - | 初始版本。工具体系由 23 收敛为 11（确定性预处理下沉管线节点、同类合并、web_search 删除、副作用异步化）；协议定为 OpenAI Tool Call JSON（对齐 answer/data.md 选型结论）；11 个工具 JSON Schema 定稿（单一契约源 config/tools/tools.json）；定义参数溯源约束、Observation 统一信封、各路由参考工具链（成本 Reward 基准）、并行调用规则、写入类风控；给出 ms-swift 训练轨迹格式与工具执行器双形态（在线 Docker / 云端 JSONL 内存索引）。 |
