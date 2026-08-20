# 本地评审意见（写给云端执行 AI）

> 评审范围：generate_dataset 全仓（scripts 9 个 + tools.json + llm_client + 根目录两脚本）。
> 评审结论：**整体质量高，架构决策成立，但存在 2 个 P0 阻断项，修复前禁止开跑 D1。**
> 按下表逐项修复，全部勾选后进入 D0 收口。

---

## P0 阻断项（必修，修完才准跑 D1）

### P0-1 密钥泄漏：`cloud_training/.env` 已被 git 跟踪

`git ls-files` 确认 `.env`（含真实 UNSPLASH_ACCESS_KEY）与 `__pycache__/` 已入库，违反任务书硬约束 4。

**修复标准**：
1. `.gitignore` 追加三行：`.env`、`__pycache__/`、`data/`
2. 执行：`git rm --cached cloud_training/.env` 与 `git rm -r --cached __pycache__`
3. 密钥已进 git 历史：若本仓库推送过任何远端（学校 git / GitHub），**必须去 Unsplash 后台重置 Access Key** 并同步更新云端 `.env`；若从未推远端，重置仍建议做（成本为零）
4. `.env` 权限 600

**验收**：`git ls-files` 输出不含 `.env` 与 `__pycache__`；`git status` 干净。

### P0-2 gen_products.py L242 运算符优先级 bug，seed 模式（默认）100% 崩溃

```python
if mode == "seed" and not DATA/"seeds".exists():   # AttributeError
```

已在本地实测复现：`not DATA/"seeds".exists()` 解析为 `not (DATA / ("seeds".exists()))`，`'str' object has no attribute 'exists'`。`py_compile`/ruff 均查不出，只有运行才炸——D1 第一条命令就会死。

**修复标准**：

```python
if mode == "seed" and not (DATA / "seeds").exists():
```

全仓 grep `not DATA/` / `DATA/"` 模式，确认无同类写法。

### P0-3 gen_products.py 种子缺失不落 pure 兜底（行为与文档矛盾）

docstring 声称「种子不可用自动落 pure」，但 `load_seed_mode` 对缺文件 `raise FileNotFoundError`、行数不足 `raise ValueError`——修完 P0-2 后，seeds 目录存在但单类目文件缺失时仍会崩。另外该函数前 10 行的首次加载是死代码（结果被 `seeds = []` 覆盖），仅起校验作用。

**修复标准**：
1. 删除 `load_seed_mode` 第一段加载循环（L121-129），校验逻辑并入第二段
2. `main()` 中调用改为 try/except：`except (FileNotFoundError, ValueError)` → 打日志、`mode = "pure"` 继续
3. pure 兜底触发时必须在汇报中说明原因与耗时预估（任务书异常表要求）

---

## P1 必修项（D0 收口前完成）

### P1-1 `hash(cat)` 种子不可复现（fetch_amazon_seeds.py L87）

`random.Random(SEED + hash(cat) % 10000)`：Python 字符串 hash 默认每进程加盐（PYTHONHASHSEED 未固定），两次运行水库抽样结果不同，违反任务书「随机种子固定、幂等可复现」。

**修复标准**：改为确定性偏移，例如 `random.Random(SEED + sum(map(ord, cat)))` 或显式 `{"3c": 1, "clothing": 2, "home": 3, "food": 4}` 映射。`map_brand` 的 `rng.choice` 同理受影响，一并覆盖。

**验收**：不设 PYTHONHASHSEED 连跑两次（删掉输出后），四份 seeds jsonl 逐字节一致。

### P1-2 CWD 相对路径依赖（四个脚本混用绝对/相对路径）

已混用实例：gen_products.py L236/258-263（`open("data/products.jsonl")`）、gen_logistics.py L56、gen_test_samples.py L95-96、train_executor.py `data_dir="data"` 默认值。同一脚本内 `DATA`（绝对）与 `"data/..."`（相对）并存——从 cloud_training/ 运行恰好一致，从其他 CWD 运行则 FileNotFoundError 或幂等判断失真。

**修复标准**：全仓统一只准用 `DATA` 常量（绝对路径）；`train_executor.py` 默认值改为模块级 `DATA`。grep 检查 `open("data`、`"data/`、`data_dir="data"` 清零。

### P1-3 requirements-cloud.txt 缺失

依赖目前散落在各 docstring。**修复标准**：按任务书 3.2 节创建于仓库根，实际用到什么写什么：

```text
httpx>=0.27
Pillow>=10.0
qrcode>=7.4
datasets>=2.19
ruff>=0.5
huggingface_hub>=0.23
llama-cpp-python>=0.3.35   # CUDA 后端:CMAKE_ARGS="-DGGML_CUDA=on" pip install ...
```

llama-cpp-python 的 CUDA 构建命令写进文件注释（无 CUDA 的默认构建会掉到 CPU，13 tok/s 变 1 tok/s，D1 直接报废）。

### P1-4 chat/task.md 是 v1.0 旧任务书

仍是 Ollama/vLLM 方案、`~/cloud-training/` 工作目录、Slurm 预案。阶段 5-8 的指引（llama-server 轨迹采样、WORKDIR 约定、归档命令、异常表）全部过时。你已实际按 llama.cpp 路线执行（llm_client 决策正确，D1-D3 进程内单流 13 tok/s 对 seed 模式 ~5h 可接受），但后续阶段不能再按旧文档走。

**修复标准**：用主仓库 `TicketAutomationPlatform/cloud-agent-taskbook.md`（v1.1）整体覆盖 `chat/task.md`，`diff` 为空。

### P1-5 pure 模式输出必然截断（兜底路径质量隐患）

`PROMPT_PURE` 一次要求 250 条商品（需 12k+ 输出 token），而 `max_tokens=4096`、`n_ctx=8192`——JSON 数组必截断，`extract_json_array` 解析失败重试 3 次后走模板兜底，产出「品质保证七天无理由退换」式垃圾描述污染 5000 SKU。Amazon 源不可达时就会走到这条路。

**修复标准**：pure 模式每批改为 20-25 条（PROMPT_PURE 的 `{n}` 与调用处同步改）；模板兜底条数占最终产物比例打印进汇报，> 30% 需上报而不是静默通过。

---

## P2 建议（不阻断，顺手修）

1. train_executor.py L349-351 死代码（`for p in ...: pass`）删除；`_tool_text_search` 命中条件 `if hits or ... and any(...)` 第二支与 `hits` 语义重复，化简为 `if hits`。
2. fetch_product_images.py：`skipped` 每个 query 重置，最终汇报只含最后一个词的失败数——改为全程累计。
3. download_model.py：阶段 5（run_teacher 带图轨迹）必须走 llama-server + mmproj（llama-cpp-python 无视觉能力）。现在就补 mmproj 下载：改用 `snapshot_download(..., allow_patterns=[f"*{quant}*.gguf", "mmproj*"])` 并在脚本尾自检两文件齐备。这是提前修，避免 D3 中断。
4. cs_reward.py 单测：consult/refund_track/adversarial 名义 case 数低于「每路由 ≥ 20」（矩阵测试部分覆盖）——各补 5-10 个边界 case 即达标。

---

## 协议约定（后续每阶段执行）

- `chat/last.md`：每阶段末由你（云端 AI）写入本阶段汇报（验收勾选 + 关键统计 + decisions 摘要），当前为空，从 D0 修复汇报开始启用
- `chat/suggestion.md`：本地评审通道（本文件），每次 push 后先读此文件再继续
- decisions.log 当前缺失：llm_client 决策引用了它却不存在于仓库——补建并回填 2026-08-19 的 llama.cpp 决策记录

---

## 修复完成定义（D0 放行 D1 的核对清单）

- [ ] `git ls-files` 不含 `.env` / `__pycache__`；`.gitignore` 含 `.env`、`__pycache__/`、`data/`
- [ ] `not (DATA / "seeds").exists()` 修复；全仓无同类优先级写法
- [ ] 种子缺失/行数不足自动落 pure（干跑验证：临时移走一个 seed 文件，确认不崩且打日志）
- [ ] fetch_amazon_seeds 删输出重跑两次，产物逐字节一致
- [ ] 全仓 grep 无相对 `data/` 路径；train_executor 默认 data_dir 用绝对常量
- [ ] requirements-cloud.txt 就位（含 llama-cpp-python CUDA 构建注释）
- [ ] chat/task.md 与主仓库 v1.1 任务书 diff 为空；decisions.log 补建
- [ ] `cs_reward.py --selftest` 全绿（已达成，回归确认）；ruff + py_compile 全过
- [ ] 以上结果写入 chat/last.md 后 push，等本地确认放行 D1
