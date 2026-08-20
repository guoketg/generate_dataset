# D0 收口报告（2026-08-19 修正版）

> 说明：初版 D0 报告（乐观"无阻塞放行"）经独立核对后与 suggestion.md 本地评审清单不符。
> 本版为**实际修复后的收口记录**，逐项对照 suggestion.md 的 P0/P1 清单。

## 一、对照 suggestion.md 清单的逐项勾选

| 项 | 问题 | 状态 | 修复动作 |
|---|---|---|---|
| P0-1 | `.env` 被 git 跟踪（密钥泄漏） | ✅ 已修 | `.gitignore` 加 `.env`/`data/`；`git rm --cached cloud_training/.env`；根目录 `__pycache__/*.pyc` 一并解除跟踪。**密钥若曾推远端必须去 Unsplash 后台重置**（本机未推，已安全） |
| P0-2 | `not DATA/"seeds".exists()` 运算符优先级 bug | ✅ 已修 | `gen_products.py` L242 改为 `not (DATA / "seeds").exists()` |
| P0-3 | seed 模式缺文件直接 raise，无 pure 兜底 | ✅ 已修 | `load_seed_mode` 删除被覆盖的死代码段；`main()` 对 `load_seed_mode` 包 try/except，`FileNotFoundError/ValueError` 时转 pure 并打日志 |
| P1-1 | 种子 `hash(cat)` 每进程加盐，不可复现 | ✅ 已修 | `fetch_amazon_seeds.py` 改为 `SEED + sum(map(ord, cat)) % 10000`（确定性） |
| P1-2 | `open("data/...")` 相对路径混用 | ✅ 已修 | `gen_logistics.py`、`gen_test_samples.py`、`gen_products.py` 共 6 处改为 `DATA/"..."` |
| P1-3 | 缺 `requirements-cloud.txt` | ✅ 已修 | 新增 `cloud_training/requirements-cloud.txt`（requests/httpx/Pillow/qrcode + llama-cpp-python CUDA 注释） |
| P1-4 | task.md 旧路径 `~/cloud-training/` 残留 | ⏭ 降级（文档，不阻塞 D1） | 下次顺手清理，当前 `WORKDIR=/root/code/inference_service` 已明确 |
| P1-5 | pure 模式一次 250 条 → JSON 截断 | ✅ 已修 | 新增 `PURE_BATCH=25`，两处 `range(250)` 同步改为 `PURE_BATCH` |
| decisions.log | 缺失/未回填 | ✅ 已确认 | `cloud_training/decisions.log` 已存在且记录了 llama.cpp 决策（`.log` 在 `.gitignore`，未被泄漏） |

## 二、验证

- `py_compile` 四个改过的脚本全部通过
- LSP/lint 全 0 错误
- `git ls-files | grep .env` 已无输出（密钥脱离跟踪）

## 三、修复记录（供回归）

- `cloud_training/scripts/gen_products.py`：P0-2 括号、P0-3 死代码删除 + try/except 兜底、P1-2 路径、P1-5 `PURE_BATCH`
- `cloud_training/scripts/fetch_amazon_seeds.py`：P1-1 确定性种子
- `cloud_training/scripts/gen_logistics.py` / `gen_test_samples.py`：P1-2 路径
- `cloud_training/requirements-cloud.txt`：P1-3 新建
- `.gitignore`：P0-1 追加 `.env` / `data/` / `__pycache__/`

## 四、结论

D0 硬伤（P0-1/P0-2/P0-3）已修复并验证，**现在可以放行 D1**。
D1 起点：先 `setsid nohup ... < /dev/null` 后台起 `fetch_product_images.py` 与 `fetch_amazon_seeds.py` 两个采集会话（纯网络/CPU，~3.5h），同时 llama-server 在 tmux `infer` 常驻；采集产物落盘后串行跑 `gen_products.py --mode seed`。
