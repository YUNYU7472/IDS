# 三节点链路协同实时入侵检测系统使用说明

简短概述：本仓库提供三段管道协同的实时入侵检测流：`lite_agent.py`（采样 → JSONL）| `kdd_mapper.py`（滑窗聚合 → 伪 KDD JSONL）| `realtime_ids.py`（加载 ckpt 推理 → 输出 CSV）。本文档以可复制命令为主，便于快速上手与排查。

---

## 快速开始（跑 50 条）

一条命令，采样 50 条并通过 mapper/IDS 完整流程（示例会产生若干 KDD 输出与最终检测结果）：

```bash
python lite_agent.py --interval 0.2 --max-rows 50 \
  | python kdd_mapper.py --model-path ckpt/model.pth --window-sec 2 --min-samples 2 --max-rows 10 \
  | python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO
```

（说明：`lite_agent` 生成 50 条采样；`kdd_mapper` 根据窗口参数输出若干条伪 KDD 样本；`realtime_ids` 将这些 JSONL 转为预测并写入 `out.csv`。）

---

## 参数与必要项（Parameters）

### 1. `lite_agent.py`（主机信息采集）

| 参数          | 是否必需 | 默认值 | 说明                                                  |
| ------------- | -------- | ------ | ----------------------------------------------------- |
| `--interval`  | 否       | `0.5`  | 采样间隔（秒），每隔多久采集一次主机信息              |
| `--max-rows`  | 否       | `0`    | 最大采样条数，`0` 表示不限制（持续运行，Ctrl+C 结束） |
| `--log-level` | 否       | `INFO` | 日志级别：DEBUG / INFO / WARNING / ERROR              |

**输出（stdout）**

- JSONL（每行一个 JSON）
- 仅输出数据，不输出日志

**最小输出结构示例：**

```

{
  "ts_ms": 1767756320609,
  "features": {
    "cpu_percent": 12.31,
    "mem_used_mb": 80709.0,
    "rx_bytes": 25798480.0
  },
  "meta": {
    "host": {...},
    "net": {...}
  }
}
```

------

### 2. `kdd_mapper.py`（特征映射 / 窗口聚合）

| 参数            | 是否必需 | 默认值 | 说明                                          |
| --------------- | -------- | ------ | --------------------------------------------- |
| `--model-path`  | **是**   | 无     | KDD 模型 checkpoint，用于读取 `feature_names` |
| `--window-sec`  | 否       | `2.0`  | 滑动窗口长度（秒）                            |
| `--min-samples` | 否       | `2`    | 窗口内最少采样数，低于则不产出                |
| `--max-rows`    | 否       | `0`    | 最大输出样本数，`0` 表示不限制                |
| `--log-level`   | 否       | `INFO` | 日志级别                                      |

**功能说明：**

- 将 `lite_agent` 的真实采样
- 按时间窗聚合
- 输出 **KDD 特征空间对齐的伪样本**

**输出（stdout）**

- JSONL
- 每行 `features` **完整覆盖模型 33 个特征**

------

### 3. `realtime_ids.py`（实时入侵检测）

| 参数           | 是否必需 | 默认值    | 说明                         |
| -------------- | -------- | --------- | ---------------------------- |
| `--input-type` | 否       | `jsonl`   | 输入类型：`jsonl` / `csv`    |
| `--model-path` | **是**   | 无        | 模型 checkpoint              |
| `--output`     | 否       | `out.csv` | 推理结果输出 CSV             |
| `--log-level`  | 否       | `INFO`    | 日志级别                     |
| `--quiet`      | 否       | 关闭      | 等价于 `--log-level WARNING` |

**输入（stdin）**

- JSONL：每行需包含 `features`，且字段与模型完全一致
- CSV：NSL-KDD 原始格式

## 常用脚本（可复制命令块）

A) 三文件协同：实时读取本机信息进行分析

- A1：跑 50 条后自动退出（示例：mapper 最多输出 10 条）：
```bash
python lite_agent.py --interval 0.2 --max-rows 50 \
  | python kdd_mapper.py --model-path ckpt/model.pth --window-sec 2 --min-samples 2 --max-rows 10 \
  | python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO
```

- A2：一直运行直到按 Ctrl+C（不限制 max-rows；建议无缓冲运行以实时性更好）：
```bash
PYTHONUNBUFFERED=1 python -u lite_agent.py --interval 0.2 \
  | python kdd_mapper.py --model-path ckpt/model.pth --window-sec 2 --min-samples 2 \
  | python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO
```

B) 仅使用 KDD 数据集调用 realtime_ids.py

- B1：JSONL 数据集跑 50 条（从文件截取前 50 行）：
```bash
head -n 50 data/KDDTest+.jsonl | python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO
```

- B2：JSONL 数据集全跑完：
```bash
python realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO < data/KDDTest+.jsonl
```

- B3：CSV 数据集跑 50 条（按 CSV 前 50 行）：
```bash
head -n 50 data/KDDTest+.csv | python realtime_ids.py --input-type csv --model-path ckpt/model.pth --output out.csv --log-level INFO
```

- B4：CSV 数据集全跑完：
```bash
python realtime_ids.py --input-type csv --model-path ckpt/model.pth --output out.csv --log-level INFO < data/KDDTest+.csv
```

- B5：模拟持续循环输入（JSONL），用于压力/长期测试（Bash）：
```bash
# 持续循环 KDDTest+.jsonl 并实时推理（注意防止缓冲）
while true; do cat data/KDDTest+.jsonl; done | PYTHONUNBUFFERED=1 python -u realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output out.csv --log-level INFO
```

（在 Windows PowerShell 中可用循环替代命令或在 WSL/Bash 环境执行上述命令。）

---

## 输出与日志解释

- `realtime_ids` 的输出 CSV (`--output`, 默认 `detection_results.csv`) 每行包含：
  - `idx`：输出序号（从 1 开始）
  - `pred_label`：数值标签（0 = 正常，1..4 = 各类攻击）
  - `pred_name`：中文标签（如 “正常”、“拒绝服务(DoS)” 等）
  - `confidence`：预测置信度（小数）
  - `latency_ms`：该条样本推理耗时（毫秒）
  - `src_ip, dst_ip, src_port, dst_port, proto`（如果输入 `meta` 中包含，则写入）
- 最终统计（运行结束时，脚本会在 stderr 打印一次）字段含义：
  - `total_rows`（或 总输入）：读入的行数 / 采样数
  - `ok_rows`（或 有效检测）：成功推理的样本数
  - `bad_rows`（或 无效行）：被跳过的行数（例如缺字段、解析失败、维度不匹配、NaN/Inf）
  - `alerts`（告警数）：`pred != 0` 的条数
  - `per_label_counts`：按类别计数（脚本会用中文标签显示）
  - `avg_latency_ms`（平均延迟）：平均单样本推理延迟（ms）
  - `throughput_rows_per_sec`（吞吐）：有效检测数 / 总时长（条/秒）
- 关于 Ctrl+C：按下 Ctrl+C 会触发 KeyboardInterrupt，程序会尽量优雅退出并打印一次最终统计，这属于正常现象且不影响已完成的输出。

---

## 故障排查（最常见 6 条）

1. 报错 “unrecognized arguments: --stdin”
   - 说明：不要传不存在的 `--stdin` 参数；正确的输入由 `--input-type` 指定为 `jsonl` 或 `csv`，并通过 stdin 传入数据。

2. 出现 BrokenPipe / traceback 来自上游输出
   - 说明：下游进程提前退出会导致上游写 stdout 时触发 BrokenPipe，`lite_agent` 与 `kdd_mapper` 已实现优雅捕获并安静退出（会在 stderr 打印中文 INFO，例如 “下游管道已关闭（BrokenPipe），kdd_mapper 停止输出并退出”）。排查：检查下游（通常为 `realtime_ids.py`）是否启动并正常工作。

3. `realtime_ids` 报 `missing_field`（或输入被计为 bad_rows）
   - 说明：输入的 JSONL 行在 `features` 字典中缺少某些模型期望的字段，`realtime_ids` 会将该行计作 `missing_field` 并跳过。排查：确认 `kdd_mapper.py` 输出的 `features` dict 覆盖 checkpoint 的所有 `feature_names`（kdd_mapper 在启动时也会打印代理/近似字段摘要）。

4. 模型加载失败 / torch.load 报参数错误（例如 weights_only）
   - 说明：可能是 torch 版本差异。解决办法：升级到较新版本的 torch，或使用脚本已加入的兼容回退（脚本中对旧版 torch 参数做了 try/except 回退）。

5. 输出中包含非 JSON 文本（影响管道）
   - 说明：stdout 必须只含 JSONL；脚本保证将日志输出到 stderr。若你看到非 JSON 内容出现在 stdout，请确认你没有在管道中把 stderr 重定向到 stdout。

6. `lite_agent` 没有 `psutil` 时采样信息缺失
   - 说明：如果系统未安装 `psutil`，`lite_agent` 会降级使用 `/proc` 采集；在某些平台上这些字段可能为 `null`/None，属于预期降级行为。

---

该文档提供可复制命令、参数说明与常见排查要点，适合复制粘贴直接运行。