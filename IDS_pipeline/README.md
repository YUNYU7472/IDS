# 五节点实时入侵检测系统（IDS Pipeline）

> 本工程实现了一套 **五节点实时入侵检测联动系统**，覆盖  
> **主机采样 → 特征映射 → 实时推理 → 响应决策 → 蜜罐诱捕**  
> 同时保留并规范化原 README 中的 **所有常用命令**，在此基础上补充工程化说明与参数表，适用于课程设计、验收与答辩展示。

---

## 一、系统总体架构（五节点）

！！！！！架构图占位！！！！！

---

## 二、节点说明与参数表

### 节点 ①：`lite_agent.py` —— 主机轻量采样

**功能**
- 周期性采集 CPU / 内存 / 网络 / TCP 状态
- 每次采样输出一行 JSON（JSONL）到 stdout
- 用于模拟真实在线主机环境

**参数表**

| 参数          | 类型  | 说明                           |
| ------------- | ----- | ------------------------------ |
| `--interval`  | float | 采样周期（秒）                 |
| `--max-rows`  | int   | 最大输出行数（不设则持续运行） |
| `--log-level` | str   | 日志级别                       |

**示例**

```bash
python lite_agent.py --interval 0.2 --max-rows 100
```

### 节点 ②：`kdd_mapper.py` —— 特征映射 / 滑窗聚合

**功能**

- 将 `lite_agent` 的 JSONL 输入按时间窗口聚合
- 映射为 KDD 风格特征 JSONL
- 确保特征维度与模型 checkpoint 完全一致

**参数表**

| 参数            | 类型  | 说明                         |
| --------------- | ----- | ---------------------------- |
| `--model-path`  | str   | 模型 checkpoint 路径（必填） |
| `--window-sec`  | float | 聚合滑窗长度（秒）           |
| `--min-samples` | int   | 窗口最少样本数               |
| `--max-rows`    | int   | 最大输出条数                 |
| `--log-level`   | str   | 日志级别                     |

**示例**

```
python lite_agent.py --interval 0.2 | \
python kdd_mapper.py --model-path ../ckpt/model.pth --window-sec 2.0 --max-rows 50
```

### 节点 ③：`realtime_ids.py` —— 实时入侵检测

**功能**

- 从 stdin 读取 JSONL / CSV
- 加载模型进行推理
- 输出检测结果 CSV
- 当 `pred != 0` 时自动联动 `response_actions`

**参数表**

| 参数           | 类型 | 说明                         |
| -------------- | ---- | ---------------------------- |
| `--input-type` | str  | 输入类型：`jsonl` / `csv`    |
| `--model-path` | str  | 模型 checkpoint 路径（必填） |
| `--output`     | str  | 输出 CSV 文件                |
| `--log-level`  | str  | 日志级别                     |
| `--quiet`      | flag | 安静模式                     |

------

## 三、常用命令

### 1）KDD 数据集前 50 条实时检测（联动示例）

```
head -n 50 ../data/KDDTest+.jsonl | \
python realtime_ids.py --input-type jsonl --model-path ../ckpt/model.pth --output out.csv --log-level INFO
```

说明：

- 输出：`out.csv`
- 当 `pred != 0` 时自动触发 `response_actions`
- 写入 `honeypot_redirect.log`
- 若蜜罐运行，会向 `127.0.0.1:8080/beacon` 发送事件通知

------

### 2）顺序循环推送 KDD 数据集（持续运行，低资源）

```
while true; do
  cat ../data/KDDTest+.jsonl
done | python realtime_ids.py \
    --input-type jsonl \
    --model-path ../ckpt/model.pth \
    --output out.csv \
    --log-level INFO
```

说明：

- 顺序读取数据集，读到结尾后自动从头开始
- `realtime_ids` 只启动一次
- 更接近真实实时检测行为

------

### 3）全链路真实采样（`lite_agent → kdd_mapper → realtime_ids`）

```
python lite_agent.py --interval 0.1 --max-rows 1000 | \
python kdd_mapper.py --model-path ../ckpt/model.pth --window-sec 2.0 --max-rows 100 | \
python realtime_ids.py --input-type jsonl --model-path ../ckpt/model.pth --output detection_results.csv --log-level INFO
```

### 4）全链路持续运行（手动中断）

```
python lite_agent.py --interval 0.5 | \
python kdd_mapper.py --model-path ../ckpt/model.pth --window-sec 2.0 | \
python realtime_ids.py --input-type jsonl --model-path ../ckpt/model.pth --output detection_results.csv
```

## 四、节点 ④：`response_actions.py` —— 响应决策模块（dry-run）

**功能**

- 根据检测结果执行响应策略
- 当前为 dry-run（不修改系统网络）
- 生成可解释重定向命令字符串
- 写入 `honeypot_redirect.log`
- 通过 HTTP beacon 通知蜜罐

**关键策略字段（ResponseConfig）**

| 字段                   | 说明             |
| ---------------------- | ---------------- |
| `confidence_threshold` | 告警置信度阈值   |
| `cooldown_sec`         | 冷却时间         |
| `label_policy`         | 攻击类型触发策略 |
| `honeypot_ip`          | 蜜罐 IP          |
| `honeypot_port`        | 蜜罐端口         |
| `mode`                 | 固定为 `dry-run` |

**本地演示**

```
python response_actions.py --demo
```

## 五、节点 ⑤：`honeypot.py` —— 低交互蜜罐

**功能**

- 监听 TCP 连接
- 接收 `response_actions` 的 beacon
- 写入 JSONL 取证日志
- 不提供真实服务，仅用于诱捕与演示

**参数表**

| 参数          | 类型 | 说明         |
| ------------- | ---- | ------------ |
| `--host`      | str  | 监听地址     |
| `--port`      | int  | 监听端口     |
| `--log`       | str  | 输出日志文件 |
| `--log-level` | str  | 日志级别     |

**启动命令**

```
python honeypot.py --host 0.0.0.0 --port 8080 --log honeypot_hits.jsonl --log-level INFO
```

## 六、推荐联合演示流程（验收级）

### 终端 A：启动蜜罐

```
python honeypot.py --port 8080
```

### 终端 B：运行 IDS（KDD 仿真）

```
head -n 50 ../data/KDDTest+.jsonl | \
python realtime_ids.py --input-type jsonl --model-path ../ckpt/model.pth --output out.csv --log-level INFO
```

