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

## 7. 项目当前状态（2026-08-21）

**⚠️ 重要：v1 训练数据已作废，禁止用于训练。**

v1 产物（`data/training/`）存在 15 个关键 bug（A-O），需按 v2 方案重做。

**详细审核报告**：`chat/2026-08-20-v2-audit.md`  
**完整任务书（含补丁代码）**：`chat/suggestion1.md`

**当前进度**：
- 已完成所有15个bug修复（A-O）
- 已完成品牌翻译（保留真实品牌名）
- 已重新生成训练数据（29966条）
- **验收状态：11/12检查通过，1项失败（多样性检查）**
- 多样性检查：48-51%（需要≥60%）

**⚠️ 关键问题（用户指出）**：
**训练数据基于模板生成，非LLM生成**：当前`gen_training_data.py`完全基于预定义模板和随机化，没有使用LLM生成自然语言查询。模型会学到模板模式，而非真实语言表达，导致泛化能力差。

**下一步（按优先级）**：
1. **解决关键问题**：修改数据生成流程，使用Qwen3.8-28B-4bit生成自然语言查询
2. 优化多轮路线多样性（当前瓶颈：multi_turn_vague 5.3%，multi_turn_ocr_logistics 10.7%）
3. 重新运行验证直到12/12通过
4. 生成最终训练数据
5. 准备模型训练

## 8. 已知坑

- `gen_products.py` 曾因 `(DATA/"x".exists())` 运算符优先级 bug 崩溃，已修复为 `(DATA/"x").exists()`。
- llama_cpp.server 依赖链需 `pydantic` + `pydantic-settings` + `starlette-context` + `sse-starlette`，venv 默认未全装，首次启动前需补。
- 单 slot server 下 `concurrency>1` 不会加速，仅快进快出；不要误以为"并发=更快"。
