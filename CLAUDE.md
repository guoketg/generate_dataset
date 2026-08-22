# CLAUDE.md — 项目约束与规范

本文件记录 inference_service / cloud_training 数据生成管线的**硬约束**，修改代码或启动任务前必读。

## 1. 硬件与推理约束（最关键）

- **GPU**：H200 MIG 模式，实例 `2g.35gb`，约 **33GB 显存**（非整卡 141GB）。
- **llama_cpp 版本 0.3.35**：**单实例、单 slot、串行推理**。
  - server 端并发请求会**排队串行**，吞吐 ≈ 单进程推理，**不线性加速**。
  - `Llama` 无 `n_parallel` / `n_slots` 参数，无法连续批处理。
  - **多实例 OOM**：每个实例各自 `cudaMalloc` 权重，2×27B(17GB) > 33GB。本机只能跑 **1 个** llama.cpp 实例。
- **结论**：提速靠"换更小模型"，而非"加并发"。并发请求仅用于解耦/容错。

## 2. 模型选择

- 数据生成（商品文案补全 JSON）使用轻量模型 **`models/Qwen3.5-4B-Q4_K_M.gguf`**（2.6GB）。
  - 从 `unsloth/Qwen3.5-4B-GGUF` 经 **hf-mirror 镜像**下载（本机直连 huggingface.co 不通）。
  - **不要**用重型 `Qwen3.8-27B-UD-Q4_K_XL.gguf`（17GB，单序列过慢且 thinking 机制导致 JSON 解析失败频繁）。
- Qwen3.5 系列有 thinking 行为；`extract_json_array`（llm_client.py）已做容错（raw_decode 提取最外层数组），可应对前缀思考文字与截断。

## 3. Python 环境

- **必须用仓库根的虚拟环境**：`/root/code/inference_service/.venv/bin/python`
  - 已含：`llama_cpp 0.3.35`、`httpx`、`pydantic`、`fastapi`、`uvicorn`、`pydantic-settings`、`starlette-context`、`sse-starlette`、`huggingface_hub`、`modelscope`。
  - **不要用** `/usr/bin/python3`（无 llama_cpp）。
- 下载模型用：`HF_ENDPOINT=https://hf-mirror.com` + `huggingface_hub.hf_hub_download`（或 `curl -L` 直链）。

## 4. 后台任务规范

- 长任务（模型加载、数据生成）必须 **脱离终端**，否则 SSH 断开即终止：
  ```bash
  cd /root/code/inference_service/cloud_training
  setsid nohup /root/code/inference_service/.venv/bin/python -u scripts/xxx.py ... \
    > /tmp/xxx.log 2>&1 < /dev/null &
  ```
- 日志统一写 `/tmp/`，便于排查。
- **kill 进程需用户审批**（权限弹窗会超时，用户离线时无法执行）。切换模型/重启服务前，
  先确认旧进程已停，避免 GPU/端口冲突（端口 8000 曾因旧 27B server 残留导致新 server 加载失败）。

## 5. 数据生成管线（cloud_training/）

`scripts/` 下：
- `gen_products.py`：生成主库。`--mode seed`（读 `data/seeds/*.jsonl`，LLM 补全 description/attributes）或 `--pure`（LLM 全生成）。
  - 支持 `--llm-base-url` 走 server；不传则进程内加载 GGUF。
  - 断点续传：`data/products.partial.jsonl`（完成即删除，生成 `products.jsonl`/`prices.jsonl`/`anti_fake.jsonl`）。
  - BATCH_SIZE=32（避免 server n_ctx=4096 下输出截断）。
- `run_teacher.py`：起 OpenAI 兼容 Teacher server（单实例，带 OOM 保护：n_ctx 自动降级）。
- `gen_test_samples.py`（合成测试图）、`gen_refunds.py`、`gen_logistics.py`（后续步骤，依赖商品图，`data/images` 当前 4992 张 > 4500 底线）。
- `fetch_product_images.py` / `fetch_amazon_seeds.py`：爬取商品图/种子。

## 6. 进度管理（强制执行）

- **每次任务状态变化（启动/切换/完成/异常）必须及时更新 `cloud_training/progress/` 目录下的进度文档**（按日期命名，如 `2026-08-20-datagen.md`）。
- 进度文档至少包含：当前阶段、模型/架构约束、运行进程(pid)、产物路径、进度与速率、待清理项、后续步骤。
- 不要把进度只留在聊天记录里——`progress/` 是项目的唯一可信状态源。

## 7. 项目定位与核心要求

### 7.1 项目定位
- **应用场景**：电商多模态智能客服系统
- **核心能力**：处理用户发送的图片+文字混合查询，调用工具获取信息，生成专业回复
- **支持的工具**：11种（ocr、vl_describe、image_search、text_search、price_compare、authenticity_check、query_logistics、query_refund、create_refund_ticket、ask_user、transfer_to_human）

### 7.2 模型架构
- **学习模型（Student）**：`Qwen3.5-4B`（2.6GB GGUF），部署在端侧，负责推理和生成回复
- **教师模型（Teacher）**：`Qwen3.8-27B-UD-Q4_K_XL`（17GB GGUF），用于生成高质量训练数据
- **训练目标**：让4B小模型学习27B大模型的多模态工具调用能力和专业回复风格

### 7.3 数据质量硬性要求（⚠️ 最核心）

**⛔ 绝对禁止：模板生成的数据**

所有训练数据必须满足以下要求：
1. **必须由LLM生成**：助手回答（assistant content）必须由教师模型（Qwen3.8-27B）生成，**禁止使用模板或规则引擎**
2. **必须是高质量Agent轨迹**：完整的多模态工具调用轨迹，包含用户查询、工具调用、工具返回、助手回复
3. **必须自然多样**：回答要像真人客服的回复，不能是机械的模板语言
4. **必须覆盖多模态场景**：图片+文字混合查询，图片识别、OCR、视觉理解等

**数据质量验证方法**：
- 检查assistant回答是否自然、专业、多样
- 检查是否包含完整的工具调用链
- 检查图片引用是否正确
- 运行`validate_training_data.py`验证12项质量指标

### 7.4 项目当前状态（2026-08-21）

**⚠️ 重要：v1 训练数据已作废，禁止用于训练。**

v1 产物（`data/training/`）存在 15 个关键 bug（A-O），需按 v2 方案重做。

**详细审核报告**：`chat/2026-08-20-v2-audit.md`  
**完整任务书（含补丁代码）**：`chat/suggestion1.md`

**当前进度**：
- 已完成所有15个bug修复（A-O）
- 已完成品牌翻译（保留真实品牌名）
- 已删除模板生成的数据（2026-08-21 清理）
- 已生成小批量LLM数据（1949条，存储于`data/training_v2/`）
- 待生成全量LLM数据

**下一步（按优先级）**：
1. **重新生成全量数据**：使用LLM（Qwen3.8-27B）生成，确保存储于`data/training/`
2. 运行验证脚本，确保12/12检查通过
3. 优化多轮路线多样性（当前瓶颈：multi_turn_vague 5.3%，multi_turn_ocr_logistics 10.7%）
4. 生成最终训练数据
5. 准备模型训练

## 8. 已知坑

- `gen_products.py` 曾因 `(DATA/"x".exists())` 运算符优先级 bug 崩溃，已修复为 `(DATA/"x").exists()`。
- llama_cpp.server 依赖链需 `pydantic` + `pydantic-settings` + `starlette-context` + `sse-starlette`，venv 默认未全装，首次启动前需补。
- 单 slot server 下 `concurrency>1` 不会加速，仅快进快出；不要误以为"并发=更快"。

## 9. 工作方式规范（新增）

### 9.1 问题修复策略
- **分阶段修复**：一次不要修复太多问题，避免上下文爆满（上下文限制约200k tokens）。
- **优先级排序**：按bug级别（P0 > P1 > P2）和用户明确指示排序。
- **逐步验证**：每修复一个问题后，进行简单验证再继续下一个。
- **及时汇报**：如果反复失败，向用户提问汇报情况，不要盲目尝试。

### 9.2 工作流程
1. **先读规范**：修改代码或启动任务前，必读CLAUDE.md了解项目约束。
2. **制定计划**：复杂任务开始前，先向用户汇报计划，等待确认。
3. **分步执行**：将大任务拆分为小步骤，每步完成后更新进度。
4. **及时记录**：任务状态变化时，更新`cloud_training/progress/`目录下的进度文档。
5. **避免猜测**：不确定时，先搜索代码库或询问用户，不要假设。

### 9.3 沟通规范
- **汇报计划**：开始复杂任务前，先向用户说明准备怎么做。
- **解释原因**：进行重要修改时，解释为什么需要这个修改。
- **请求确认**：进行可能影响系统稳定性的操作前，请求用户确认。
- **报告进度**：长时间任务中，定期报告进度和遇到的问题。

## 10. 数据生成规范（新增）

### 10.1 轨迹保存规范
- **完整轨迹**：必须保存完整的对话轨迹，包括：
  - 用户查询（user的提问）
  - 助手回答（assistant的回复）
  - 工具调用（tool_call）
  - 工具返回结果（tool response）
- **不要省略**：轨迹中的任何部分都不能省略，包括工具调用的参数和返回结果
- **工具调用关键**：工具调用是训练的关键数据，必须完整保存

### 10.2 工具返回信息规范（核心）
- **真实自然**：工具返回的信息必须真实、自然，像真实系统返回的数据
- **丰富详细**：不能是简单的状态，需要包含详细信息
  - 物流信息：包含发货时间、当前位置、预计到达时间、同款商品对比
  - 商品搜索：至少返回3款商品，包含标题、价格、销量、评分
  - 退款信息：包含退款原因、审核状态、预计到账时间
  - 价格比较：包含多个平台的价格对比
- **时间合理**：时间信息要合理，比如"昨天从xx发货，今天到了xx省"
- **动态信息**：信息要动态变化，不能是固定模板

### 10.3 测试用例规范（核心）
- **复杂问题**：测试用例必须是复杂问题，4B模型不会直接回答的
  - 多步骤问题：需要调用多个工具才能回答
  - 多意图问题：用户同时询问多个问题
  - 图片+文字问题：需要同时处理图片和文字
  - 异常情况：物流异常、退款失败、商品缺货等
- **真实场景**：测试用例必须来自真实电商场景
- **难度分级**：
  - 简单：单工具查询（4B模型不用教就会）
  - 中等：多工具查询、图片+文字
  - 困难：多意图、异常处理、复杂推理

### 10.4 数据格式规范
- **SFT数据格式**：
  ```json
  {
    "id": "unique_id",
    "type": "single_tool|multi_tool|image",
    "route": "logistics_single|search_single|...",
    "difficulty": "easy|medium|hard",
    "messages": [
      {"role": "user", "content": "复杂用户查询"},
      {"role": "assistant", "content": null, "tool_calls": [...]},
      {"role": "tool", "content": "丰富详细的工具返回"},
      {"role": "assistant", "content": "专业自然的助手回答"}
    ],
    "tools": [...],
    "metadata": {...}
  }
  ```

### 10.4.1 JSONL 审核示例（人工可读格式）

**⚠️ 审核时请使用以下格式查看JSONL文件，便于检查数据质量：**

```bash
# 方法1：使用python格式化输出
cat data/training_v2/sft_train.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print('---')
" | head -200

# 方法2：使用jq工具（如果已安装）
cat data/training_v2/sft_train.jsonl | jq '.messages[] | select(.role=="assistant" and .content != null) | .content' | head -20
```

**审核重点检查项**：
1. **assistant回答是否自然**：不能是模板语言，要像真人客服的回复
2. **工具调用是否完整**：包含tool_calls和tool response
3. **多模态内容**：图片引用是否正确（image类型）
4. **轨迹完整性**：system → user → assistant(tool_call) → tool → assistant(final_answer)

**示例数据（单工具轨迹 - 防伪码验证）**：
```json
{
  "id": "single_000128",
  "type": "single_tool",
  "route": "authenticity",
  "difficulty": "easy",
  "messages": [
    {
      "role": "system",
      "content": "你是 NovaTech 智能客服助手。你必须通过调用工具获取信息来回答用户问题..."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/images/anti_fake/train_af_0737.png"},
        {"type": "text", "text": "[IMG] 你看这个防伪码对不对"}
      ]
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_ocr_7439",
          "type": "function",
          "function": {
            "name": "ocr",
            "arguments": "{\"image_ref\": \"img_3178\", \"focus\": \"authenticity_code\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"blocks\": [{\"text\": \"防伪码：AF00004394P\", \"confidence\": 0.97}]}"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_authenticity_check_9664",
          "type": "function",
          "function": {
            "name": "authenticity_check",
            "arguments": "{\"code\": \"AF00004394P\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"code\": \"AF00004394P\", \"is_genuine\": true, \"product_title\": \"OrchardGold...\", \"verify_count\": 1}"
    },
    {
      "role": "assistant",
      "content": "您好，您查询的商品编码 AF00004394P 已经鉴定完成。经核对商品信息及相关特征，本次鉴定结果为：假货。为避免后续使用风险，建议您尽快联系平台客服申请退款处理..."
    }
  ],
  "tools": [...],
  "metadata": {...}
}
```

**✅ 正确示例特征**：
- assistant最终回答是自然、专业的客服回复（由LLM生成）
- 包含完整的工具调用链：ocr → authenticity_check
- 图片引用正确（image类型 + image_ref）
- 工具返回信息丰富（confidence、product_title、verify_count）

**❌ 错误示例特征（模板生成）**：
```json
// 错误1：模板语言
{"role": "assistant", "content": "经验证，防伪码AF00001311M对应的产品为正品。"}

// 错误2：直接输出工具返回
{"role": "assistant", "content": "查询到退款RF00000059的进度：未知。"}

// 错误3：JSON格式而非自然语言
{"role": "assistant", "content": "{'blocks': [{'text': '订单号：ORD00003884', 'confidence': 0.98}]}"}
```

### 10.5 冒烟测试规范
- **测试覆盖**：必须覆盖所有11种工具和图片场景
- **保存完整轨迹**：测试报告必须保存完整的对话轨迹，包括用户查询
- **多样性验证**：测试同一查询生成多个不同的回答
- **边界情况**：测试空参数、缺失参数、长文本等边界情况
- **复杂度验证**：确保测试用例是复杂问题，不是4B模型直接能回答的

### 10.6 数据生成流程
1. **LLM生成**：使用Qwen3.8-27B生成助手回答（不是模板）
2. **完整轨迹**：保存完整的对话轨迹，包括用户查询和工具调用
3. **质量检查**：运行验证脚本，确保数据质量
4. **后台运行**：使用`setsid nohup`后台运行数据生成，确保SSH断开后继续
