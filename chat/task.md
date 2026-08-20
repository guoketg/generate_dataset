# H200 MIG 云端执行任务书

| 字段 | 值 |
|---|---|
| version | v1.1（适配 llama.cpp 推理：替换 Ollama/vLLM 数据生成方案；工作目录改为用户常用目录；长占确认） |
| updated_at | 2026-08-19 |
| 适用环境 | 学校 H200 MIG（33 GB VRAM 单实例），SSH 远程执行 |
| 设计依据 | [prd/17-data-generation.md](prd/17-data-generation.md) v2.0、[prd/18-training-data.md](prd/18-training-data.md) v1.0、[prd/06-function-calling.md](prd/06-function-calling.md) v1.0 |
| 总窗口 | 11 天（D0-D11，暑期结束归还） |

> 本文档是交给云端 Agent 的**作业标准**：按阶段推进、按验收交付、按契约汇报。设计细节以三份 PRD 为准，本文档只定执行顺序、命令、验收与异常处置。

---

## 0. 执行约定与硬约束

**工作约定**：
- 工作目录：**用户常用目录**（用户要求能直接看到代码，不放根目录式隐蔽路径）。首次汇报必须报告实际工作目录；脚本/配置/数据必须同根（相对路径依赖）。
- 按阶段推进：D0 → D11。**D3 / D6 / D11 三个归档点必须暂停**，向用户汇报并确认 tar 已回传本地后才继续。
- 所有脚本：随机种子固定 42、幂等可重跑、进度与失败行号打印到 stdout（重定向日志文件，禁止 print 写文件）。
- 长任务一律跑在 `tmux` 会话内（防 SSH 断连），会话名按阶段命名：`infer`、`datagen`、`train`。

**硬约束（违反任意一条即任务失败）**：
1. **零付费 API**：唯一 LLM 为本地推理的 `qwen3.8:27b`（数据生成）与 `Qwen/Qwen3.5-VL-4B`（训练）。
2. **不爬取真实电商平台**（淘宝/京东/亚马逊页面）。真实感种子仅限 Amazon Reviews 2023 学术开源元数据。
3. **thinking 模式全程关闭**（原因：训练-推理格式一致性 + 批量成本 + GRPO rollout 预算 + 小模型 overthinking）：推理走服务器现有 llama.cpp `llama-server`（OpenAI 兼容 API），请求体加 `"chat_template_kwargs": {"enable_thinking": false}`；该版本不支持时兜底 `/no_think` 软开关 + `response_format: {"type": "json_object"}` 双保险；仍混入则在解析层剥离 `<think>...</think>`。D0 冒烟必须验证响应无 `<think>` 段。
4. **密钥不入代码、不入 git**：`UNSPLASH_ACCESS_KEY` 仅写 `<工作目录>/.env`（用户提供，权限 600）。
5. 任一时刻 GPU 上**只加载一个大模型**（27b 生成相位与 4b 训练相位严格串行；SFT 启动前先停 llama-server 释放显存）。
6. 遇本任务书未覆盖的决策：选保守方案、记录到 `<工作目录>/decisions.log`、汇报时说明，不擅自扩大范围。

---

## 1. 阶段总览

| 阶段 | 天 | 内容 | 检查点 |
|---|---|---|---|
| 环境预检 + 代码资产 | D0 | GPU/网络/磁盘预检；编写全部脚本与 tools.json | 预检全绿；脚本静态检查通过 |
| 业务数据生成 | D1-D3 | 商品库/比价/防伪/物流/退款 + 图片采集 + PIL 合成图 | **归档点 D3** |
| 训练数据构造 | D3-D6 | SFT 20k 三源 + GRPO 8k + 评测集 1000 + 对抗池 3600 | **归档点 D6**（含用户确认节点） |
| SFT 训练 | D6-D8 | LoRA 2 epoch + 评测跑分 | D8 扩量决策点 |
| GRPO | D8-D11 | 1k 子集闭环 → 全量 8k 迭代 | reward 曲线收敛 |
| 导出收尾 | D11 | LoRA merge → GGUF → 冒烟 → 全产物归档 | **归档点 D11** |

超支兜底序（按序触发）：砍扩量（SFT 20k 定稿）→ 砍 GRPO 轮数（保 ≥1 轮全量）→ GRPO 迁自租 5090。硬保底：D6 的 SFT checkpoint + 数据 tar 落袋。

---

## 2. D0 环境预检

按序执行，任一项失败先走对应预案再重试，连续失败即停下汇报。

```bash
# 1. GPU 可见性：应看到 MIG 设备（33 GB）
nvidia-smi -L
nvidia-smi   # driver >= 550、CUDA >= 12.4

# 2. 磁盘：可用空间 >= 60 GB
df -h ~

# 3. Python >= 3.10 与依赖（写 requirements-cloud.txt 见 3.2）
python3 --version && pip install -r requirements-cloud.txt

# 4. 网络三连测
curl -sI "https://api.unsplash.com/photos/random?client_id=$UNSPLASH_ACCESS_KEY" | head -1   # 期望 200 或 401（Key 有效时 200）
curl -sI https://hf-mirror.com | head -1                                                     # 期望 200/301
# Amazon Reviews 2023 元数据源可达性（McAuley Lab 或镜像，见 prd/17 5.3.2）

# 5. 使用服务器现有 llama.cpp（tmux 会话 infer 内启动；不装 Ollama）
llama-server --version                  # 探明版本与二进制位置
ls <部署目录>/models/                    # 确认 Qwen3.8-27B GGUF 与 mmproj（视觉投影器）是否就位
# 模型缺失则从 hf-mirror 下载 Q4_K_M GGUF（约 17 GB）+ 对应 mmproj GGUF
CUDA_VISIBLE_DEVICES=MIG-<nvidia-smi -L 查到的 UUID> llama-server \
  -m <qwen3.8-27b-Q4_K_M>.gguf --mmproj <mmproj>.gguf \
  --jinja --ctx-size 32768 --parallel 8 --cont-batching --port 8080
```

**D0 验收清单**：
- [ ] MIG 33 GB 设备可见，driver/CUDA 达标
- [ ] 磁盘 ≥ 60 GB
- [ ] Unsplash / hf-mirror / Amazon 种子源至少两个可达（全不可达立即汇报）
- [ ] llama-server 就绪（:8080），**三项冒烟全过**：(a) `enable_thinking=false` 生效，响应无 `<think>` 段；(b) 带 `tools` 的请求返回合法 `tool_calls`（验证 `--jinja`）；(c) 带图请求（base64 image_url）正常描述图片（验证 mmproj）。任一失败走异常表
- [ ] `.env` 已写入 `UNSPLASH_ACCESS_KEY`（权限 600）

---

## 3. 代码资产编写（D0 内完成，先写后跑）

全部脚本放 `~/cloud-training/scripts/`，写完只做静态检查（`ruff check .` + `python -m py_compile scripts/*.py`），**不运行**。

### 3.1 config/tools/tools.json

从 `prd/06-function-calling.md` 5.3.1-5.3.11 抄录 11 个工具 Schema，组装为 JSON 数组文件。这是工具契约单一源，后续执行器、出题器、训练注入全部读它。

### 3.2 requirements-cloud.txt

```text
httpx>=0.27
Pillow>=10.0
qrcode>=7.4
datasets>=2.19        # 公开 QA 下载（走 HF_ENDPOINT=https://hf-mirror.com）
ruff>=0.5
```

（云端环境不受本地「禁 torch」约束，但数据脚本本身也不需要 torch。）

### 3.3 业务数据脚本（6 个，prd/17 已有代码骨架，按骨架补全）

| 脚本 | 规格 | PRD 依据 |
|---|---|---|
| `fetch_amazon_seeds.py` | 流式解析 Amazon Reviews 2023 元数据（bz2/json.gz 流式，不全量下载）；每类目采样 ~1300 条（title/price/category）；真实品牌正则替换为虚构品牌表；价格按类目 1%/99% 分位裁剪；输出 `data/seeds/*.jsonl` | 17 5.3.2 种子路径 |
| `gen_products.py` | 双模式：`--mode seed`（种子改写 + 27b 仅补全 description/attributes）与 `--mode pure`（纯 LLM 兜底）；输出 products/prices/anti_fake 三份 JSONL；调 llama-server `/v1/chat/completions`（`response_format: json_object` + 关 thinking）；`extract_json_array` 容错提取 | 17 5.3.2（代码骨架已给全） |
| `gen_logistics.py` | 读 products.jsonl，每 SKU 一单，4-6 轨迹点覆盖 5 状态 | 17 5.3.3（骨架已给全） |
| `gen_refunds.py` | 5 状态 × 100 = 500 行 | 17 5.3.4（骨架已给全） |
| `gen_test_samples.py` | PIL 合成订单截图/防伪码图/瑕疵图；**参数化 `--n-e2e 20 --n-train 800`**（PRD 骨架中 N_E2E/N_TRAIN 为占位，必须实现为 argparse 参数） | 17 5.3.5 |
| `fetch_product_images.py` | Unsplash 官方 `/search/photos`；Access Key 读 `.env`；429 自动等下一小时窗口；断点续传游标落盘 | 17 5.3.6（骨架已给全） |

### 3.4 训练数据脚本（7 个，按本表规格编写）

| 脚本 | 职责与规格 | 验收 |
|---|---|---|
| `gen_questions.py` | 模板出题器。内置题干模板库：每路由 ≥ 15 种句式 × 槽位变体（单号/金额/平台/状态词随机填充）。读 17 资产 JSONL，输出构造题池：`{id, route, difficulty, messages(含图), gold, n_ref, gold_actions, split}`。`split` 字段做 train/eval 资产分区（product_id/order_id 空间互斥）。`--adversarial` 模式按 prd/18 5.5 十类配比生成对抗池 `data/adversarial/pool.jsonl`（3600 条，类目量按 18 5.5 表） | 金标随题生成且 100% 机器可判；train/eval 分区无交集（脚本自检断言） |
| `teacher_questions.py` | Teacher 出题：调 27b 按场景卡（路由组合 + 资产采样）出题，要求链长 ≥ 3、跨 ≥ 2 路由；模板校验器验证 gold 可判后入库 | 3000 题，校验通过率 > 80% |
| `run_teacher.py` | Teacher 轨迹采样：调 llama-server（`--parallel 8 --cont-batching` 已提供服务端并发，httpx 异步客户端并发 8-16）。注入 system（prd/06 5.11 模板）+ tools.json + user 题（含图）→ 采 tool_calls → `train_executor.py` 真实执行回填 Observation → 循环至终答（MAX_TOOL_LOOP=5）。每题 2 候选（temperature 0.7）。**依赖 D0 冒烟 (b) 通过**（llama.cpp 原生 tool call） | 采样 ~12k，单条轨迹 ≤ 8k tokens |
| `train_executor.py` | 训练态工具执行器（双形态之训练态，规格见 prd/06 5.10）：读 17 全部 JSONL 建内存索引（< 50 MB），实现 11 工具真实执行语义；Observation 信封按 prd/06 5.4；参数溯源校验按 prd/06 5.5 | 每工具 ≥ 3 个手工用例通过（正常/参数缺失/溯源违规拒绝） |
| `filter_trajectories.py` | 四道闸（prd/18 5.2.4）：格式（JSON/工具名∈11/Schema/tool_call_id 配对）→ 溯源 → 金标（复用 cs_reward 判分，R_answer ≥ 0.5 且 R_format = 1）→ 去重与语言（MinHash、中文题中文答、≤ 8k tokens）；输出过滤统计（各闸剔除量、yield、路由分布） | yield 与剔除统计落盘 `data/sft/filter_stats.json` |
| `build_eval.py` | 评测集组装：业务×难度矩阵（prd/18 5.6：售前 200/物流 230/退款 230/防伪 160/对抗 180；单跳 400/双跳 350/三跳+ 250）；文件头含 version/date/题目哈希；与训练题资产分区互斥 | `data/eval/eval_v1.0.jsonl` 1000 行，分区互斥断言通过 |
| `cs_reward.py` | 规则判分：完整实现 prd/18 5.4 判分表（R_format/R_answer/R_process/R_cost = 0.2/0.5/0.2/0.1，逐路由金标判分 + 对抗特殊判分）。双入口：ms-swift `external:cs_reward` 注册 + CLI 离线判分。**单测**：每路由 ≥ 20 case（部分正确/编造/超长/空答边界） | 单测全绿；CLI 对样例轨迹输出符合手算期望 |

**3 节总验收**：
- [ ] `config/tools/tools.json` 含 11 工具，`json.loads` 校验通过
- [ ] 13 个脚本 `ruff check` + `py_compile` 全部通过
- [ ] `cs_reward.py` 单测通过（唯一允许运行的脚本，纯 CPU 秒级）

---

## 4. D1-D3 业务数据生成（prd/17）

执行序（GPU 任务与 CPU/IO 任务并行排布）：

```text
tmux: infer      → llama-server 持续运行（27b 已加载，--parallel 8 --cont-batching）
tmux: datagen-1  → fetch_product_images.py（纯网络/CPU，3.5h 量级，与 GPU 任务并行）
tmux: datagen-2  → fetch_amazon_seeds.py（网络下载，并行）
之后串行：
  gen_products.py --mode seed        # 种子不可用自动落 --mode pure
  gen_logistics.py && gen_refunds.py # 分钟级
  gen_test_samples.py --n-e2e 20 --n-train 800
```

**阶段验收**（量化，全部满足才进归档）：
- [ ] `products.jsonl` = 5000 行；品牌 ∈ 虚构品牌表（脚本断言）；价格分布按类目合理
- [ ] `prices.jsonl` = 20000 行；`anti_fake.jsonl` = 5000 行（假货 = 250）
- [ ] `logistics.jsonl` = 5000 行；`refunds.jsonl` = 500 行（5 状态 × 100）
- [ ] 商品图 ≥ 4500 张（缺口 ≤ 10%，断点续传可补）；合成图三类各 820 张
- [ ] 全程零付费 API、零真实平台爬取

**D3 归档**（必停点）：

```bash
cd <WORKDIR> && tar czf cs_dataset.tar.gz data/ scripts/ config/ requirements-cloud.txt
# 汇报 tar 路径与体积（预期 <= 1 GB，不含 800/类训练合成图时约 500 MB）
# 等待用户确认已下载回本地，才进入阶段 5
```

---

## 5. D3-D6 训练数据构造（prd/18 5.2/5.3/5.5/5.6）

执行序：

```text
1. 公开 QA 下载（CPU/网络）：hf-mirror 采 OK-VQA/GQA/DocVQA/TextVQA/ChartQA 共 8k
   → 5k 纯 VQA 直接转 messages；3k 工具化改造（合成一次 ocr/vl_describe 调用）
2. gen_questions.py（构造题池：常规 4800 + 预留题池给 Teacher + 对抗池 3600）
3. 电商构造 6000 轨迹：构造题 → train_executor 真实执行 → 金标模板终答（句式随机化）
4. llama-server 起 27b 服务（tmux: infer，`--parallel 8 --cont-batching`）→ run_teacher.py 采样 ~12k → filter 留 6000
   若 D0 冒烟 (b) 未过（tool call 不可用）：按异常表升级 llama.cpp → 仍不行装 vLLM 仅服务轨迹采样，顺延 ≤ 0.5 天
5. teacher_questions.py 高难题 3000 + 构造多跳 3000 + 对抗陷阱 2000 → data/rl/prompts.jsonl
6. build_eval.py → 导出评测候选题 → 【用户确认节点】高风险/对抗题 100% 过目，
   普通题抽检 20-30%，歧义题剔除补位 → eval_v1.0.jsonl 定稿
```

**阶段验收**：
- [ ] SFT 20k = 公开 8k + 电商构造 6k（含对抗正确行为 1200）+ Teacher 6k；四道闸后格式合法率 100%、溯源违规 0、金标一致率 ≥ 95%
- [ ] `swift sft --dataset` 对三份 SFT jsonl 校验通过（含多模态样本）
- [ ] GRPO 8k 金标 100% 机器可判；对抗池 3600 十类覆盖（配比按 prd/18 5.5 表）
- [ ] 评测集 1000 矩阵符合 prd/18 5.6；**用户确认完成**（此为硬性停点）
- [ ] 过滤统计（yield/路由分布）已写入汇报

**D6 归档**（必停点）：`tar czf cs_training_data.tar.gz data/ scripts/ config/`，确认回传后进入 SFT。

---

## 6. D6-D8 SFT 训练（prd/18 5.7）

```bash
# 先停 llama-server 释放显存（硬约束 5）；训练环境由 ms-swift 管理（自带 torch / vLLM rollout），与 llama.cpp 无关
swift sft \
  --model Qwen/Qwen3.5-VL-4B \
  --dataset data/sft/public_qa.jsonl data/sft/ecommerce.jsonl data/sft/trajectories.jsonl \
  --torch_dtype bfloat16 \
  --lora_rank 16 --lora_alpha 32 --lora_target_modules all-linear \
  --num_train_epochs 2 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_ratio 0.03 --max_length 8192 \
  --gradient_checkpointing true --save_steps 500
```

- 评测：eval_v1.0 跑分，产出 baseline（原版 4b）与 SFT 版两列对比（GRPO 版列留空）。
- 超参仅做小调（lr/epoch），不预设最优；每轮调整记录到 `decisions.log`。
- **D8 决策点（需用户参与）**：整体进度富余 ≥ 1 天且 SFT 指标达标 → 评估扩量 20k→50k（优先级：电商构造 > Teacher 轨迹 > 公开 QA 固定不动）；否则 20k 定稿直接 GRPO。

**验收**：SFT checkpoint 就位；评测任务成功率显著高于原版 baseline；训练曲线无发散。

---

## 7. D8-D11 GRPO（prd/18 5.7）

```bash
# 第一步：1k 题子集先跑通闭环（reward 正常、loss 下降），通过后才放全量
swift rlhf --rlhf_type grpo \
  --model Qwen/Qwen3.5-VL-4B --resume_from_checkpoint output/sft_ckpt \
  --dataset data/rl/prompts.jsonl \
  --reward_funcs external:cs_reward \
  --num_generations 8 --temperature 1.0 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --max_length 8192 --max_completion_length 4096 \
  --vllm_mode colocate --gpu_memory_utilization 0.85
```

- OOM 处置序：`num_generations` 8→4 → `max_completion_length` 4096→2048 → 关 colocate 改 sleep 模式。
- 全量 8k 迭代 2-3 轮；每轮输出 reward 分布统计（均值/方差/零奖励占比）。

**验收**：≥ 1 轮全量迭代完成；reward 曲线收敛趋势可见；评测三列对比（原版/SFT/GRPO）产出。

---

## 8. D11 导出与最终归档

```bash
# 1. LoRA merge
swift export --ckpt_dir output/grpo_ckpt --merge_lora true
# 2. 转 GGUF Q4_K_M（ms-swift 自带 export 或 llama.cpp 转换）
# 3. 云端 CPU Ollama 冒烟：加载 GGUF，发 3 条中文题（含 1 图），验证 11 工具可调用、终答中文
# 4. 全产物归档
tar czf cs_final.tar.gz output/ data/ scripts/ config/ decisions.log
```

**验收**：GGUF 导出成功；冒烟 3/3 通过；`cs_final.tar.gz`（预期 6-8 GB）回传确认。

---

## 9. 汇报契约

每阶段完成必报（不要求过程汇报，但以下节点必须停顿等确认）：

| 节点 | 必报内容 | 是否停顿 |
|---|---|---|
| D0 完成 | 预检清单逐项勾选结果 + 冒烟输出 | 停，等用户放行 D1 |
| D3 归档 | tar 路径/体积 + 阶段验收勾选 + 生成耗时统计 | 停，等回传确认 |
| D6 归档 | 数据统计（20k/8k/1000/3600 分布）+ yield + 评测候选题导出 | 停，等用户确认评测集 |
| D8 决策 | SFT 评测两列对比 + 扩量建议 | 停，等用户决策 |
| D11 归档 | 三列评测对比 + reward 曲线 + 冒烟结果 + tar 信息 | 停，等回传确认 |
| 异常（预案外） | 现象 + 已尝试动作 + 影响评估 + 建议 | 停，等用户决策 |

---

## 10. 异常速查表

| 异常 | 处置 |
|---|---|
| Unsplash 429 配额尽 | 脚本内置等下一小时窗口；缺口 > 10% 同类目图复用补位 |
| Unsplash/网络源学校不可达 | 本地拉取后 scp 上传（图片下载不占 GPU） |
| Amazon 种子下载/解析失败 | `gen_products.py --mode pure` 纯 LLM 兜底（多耗 8-15h，需汇报顺延） |
| 27b 输出非 JSON | `extract_json_array` 容错 + 重试 3 次；仍失败记录失败行号继续 |
| thinking 拖慢单条耗时翻倍 | 检查 `chat_template_kwargs` 是否生效（硬约束 3）；老版本改用 `/no_think` + json grammar |
| llama.cpp tool call 不工作（`--jinja` 未生效/版本老） | 依序：确认 `--jinja` 已加 → 升级 llama.cpp → 装 vLLM 仅服务轨迹采样（出题与结构化生成不受影响，走 `response_format`） |
| 响应混入 `<think>` 段 | `chat_template_kwargs` 不被支持 → `/no_think` 软开关 + json grammar 双保险 → 解析层剥离 |
| mmproj 缺失 / 多模态不可用 | 从 hf-mirror 下载对应 mmproj GGUF；仍不可用则轨迹采样换 vLLM/Ollama 兜底 |
| Teacher yield < 30% | 检查 system 模板注入；每题 3 候选；仍低提高电商构造占比并汇报 |
| GRPO colocate OOM | 按 7 节处置序降配 |
| cs_reward 与离线判分不一致 | 以单测为准修 reward；已训数据不回滚 |
| 窗口整体超支 | 按 1 节兜底序执行并汇报 |
| SSH 断连 | 服务器可长期占用（用户已确认）；所有服务与长任务仍须 tmux 内跑，断连后 `tmux attach` 恢复，脚本幂等支持断点续跑 |

---

## 11. 目录契约（终态，`<WORKDIR>` 为用户常用目录，D0 首次汇报确认）

```text
<WORKDIR>/
├── config/tools/tools.json
├── data/
│   ├── seeds/ products.jsonl prices.jsonl anti_fake.jsonl logistics.jsonl refunds.jsonl
│   ├── sft/ public_qa.jsonl ecommerce.jsonl trajectories.jsonl filter_stats.json
│   ├── rl/prompts.jsonl
│   ├── eval/eval_v1.0.jsonl
│   ├── adversarial/pool.jsonl
│   └── images/ products/ orders/ anti_fake/ defects/
├── scripts/（13 个 .py）
├── output/（SFT/GRPO checkpoint + merged + GGUF）
├── decisions.log
└── .env（UNSPLASH_ACCESS_KEY，权限 600）
```
