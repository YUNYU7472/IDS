# IDS 系统测试说明文档

本文档旨在指导用户如何部署、运行并测试混合型入侵检测系统（Hybrid IDS）。该系统包含两部分核心能力：
1. **流量异常检测 (Flow-based IDS)**：基于深度学习模型，实时分析网络流量统计特征（如 DoS、Probe）。
2. **蜜罐与 AI 分析 (Honeypot + AI)**：基于低交互蜜罐捕获攻击 Payload，并利用 DeepSeek LLM 进行意图分析。

---

## 1. 环境准备 (Prerequisites)

Ensure you are in the project root directory (e.g., `/home/aet/IDS/IDS`).

### 1.1 依赖安装
确保已安装 Python 3.8+ 及必要的库（包含 `requests` 用于 AI Agent）：
```bash
pip install -r admin_train/requirement.txt
```

### 1.2 模型文件检查
检测流水线需要预训练模型。请确认 `ckpt/` 目录下存在 `model.pth`。如果不存在，请从训练目录复制：
```bash
mkdir -p ckpt
cp admin_train/best_model_20260107_103522.pth ckpt/model.pth
```

### 1.3设置 API Key (用于 AI 分析)
如果不使用 AI 分析功能可跳过此步。
```bash
export DEEPSEEK_API_KEY="your_actual_api_key_here"
```

---

## 2. 启动步骤 (Running the System)

本系统由多个组件协同工作，建议打开 **3 个独立的终端窗口** 来运行。

### 终端 A：主要检测流水线 (Main Pipeline)
该流水线包含三个模块：
1. `lite_agent`: 监控网卡流量并生成统计数据。
2. `kdd_mapper`: 将原始统计映射为 KDD99 特征格式。
3. `realtime_ids`: 加载 MLP 模型进行实时推断。

**启动命令：**
```bash
python IDS_pipeline/lite_agent.py --interval 0.5 \
  | python IDS_pipeline/kdd_mapper.py --model-path ckpt/model.pth --window-sec 2 \
  | python IDS_pipeline/realtime_ids.py --input-type jsonl --model-path ckpt/model.pth --output detection_results.csv
```
> **现象**：启动后，终端会显示 "Waiting for input..."，随后每隔几秒输出检测结果（如 "最终统计..." 或 "告警..."）。

### 终端 B：蜜罐服务 (Honeypot)
启动低交互 Web 蜜罐，监听 8080 端口，模拟易受攻击的服务。

**启动命令：**
```bash
python IDS_pipeline/honeypot.py --port 8080 --log honeypot_events.jsonl
```
> **现象**：显示 "Honeypot HTTP server listening on 0.0.0.0:8080"。

### 终端 C：AI 分析 Agent (Analysis Agent)
监控蜜罐日志，调用大模型分析攻击意图。

**启动命令：**
```bash
# 确保已设置 API Key
export DEEPSEEK_API_KEY="sk-0571f32d92c542a68dde0915d4124a40"
python IDS_pipeline/analysis_agent.py --log honeypot_events.jsonl --interval 10
```
> **现象**：显示 "Monitoring log file: honeypot_events.jsonl"。当有攻击发生时，会输出 "Analyzing X new events..." 并生成 Markdown 报告。

---

## 3. 测试与验证 (Testing)

打开 **第 4 个终端** 进行攻击模拟。

### 3.1 场景一：SQL 注入与 WebShell 攻击 (触发蜜罐与 AI)
使用 `curl` 向蜜罐发送恶意 Payload。
*注意：在本地测试时，务必加上 `--noproxy "*"` 以避免代理配置导致连接失败。*

**模拟 SQL 注入 (尝试绕过登录)：**
```bash
curl --noproxy "*" -X POST http://127.0.0.1:8080/login.php \
     -d "user=admin&pass=' OR 1=1; --" \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0) attack-tool/1.2"
```

**模拟远程命令执行 (RCE/WebShell)：**
```bash
curl --noproxy "*" http://127.0.0.1:8080/uploads/shell.php?cmd=cat%20/etc/passwd
```

**验证结果：**
1. **终端 B (蜜罐)**：会显示接收到的请求日志。
2. **终端 C (AI)**：约 10 秒后，会输出分析结果，生成类似 `analysis_report_20260108_xxxx.md` 的文件。查看该文件可以看到 AI 对 "SQL注入" 和 "敏感文件读取" 的解释。
3. **文件查看**：
   ```bash
   cat honeypot_events.jsonl  # 查看原始日志
   ls -l analysis_report_*.md # 查看生成的报告
   ```

### 3.2 场景二：流量异常检测 (触发 Flow IDS)
由于 `lite_agent` 监控的是本机全局流量，上述 `curl` 请求产生的流量特征（包数、字节数）也会被流水线捕获。
虽然单次 `curl` 很难触发 "DoS" 告警，但可以通过高频请求模拟。

**模拟高频扫描 (DoS/Probe)：**
```bash
# 快速发送 50 次请求
for i in {1..50}; do curl --noproxy "*" -s http://127.0.0.1:8080/ > /dev/null & done; wait
```

**验证结果：**
1. **终端 A (流水线)**：观察输出日志，可能会出现黄色警告信息：
   ```text
   告警 序号=XX 类型=探测(Probe) 置信度=99.12% 延迟=1.23ms ...
   ```
2. **结果文件**：
   ```bash
   tail -f detection_results.csv
   ```
   检查 `pred_name` 列是否有非 "正常" 的记录。

---

## 4. 故障排查 (Troubleshooting)

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| **蜜罐收不到请求** | 环境变量 `http_proxy` 干扰 | 使用 `curl --noproxy "*"` 测试 |
| **Pipeline 报错 FileNotFoundError** | 模型文件缺失 | 检查 `ckpt/model.pth` 是否存在 (参考 1.2) |
| **AI Agent 报错 401 Unauthorized** | API Key 无效或未设置 | 检查 `export DEEPSEEK_API_KEY` 是否正确 |
| **Pipeline 全是 "无效行" 或 NaN** | 数据归一化异常 | 代码已修复自动清洗 NaN/Inf，确保使用最新版 `realtime_ids.py` |

---

## 5. 文件结构说明

*   `IDS_pipeline/`
    *   `realtime_ids.py`: 核心检测引擎 (MLP 模型推断)
    *   `honeypot.py`: 蜜罐服务 (数据采集)
    *   `analysis_agent.py`: AI 分析服务 (DeepSeek 接口)
    *   `lite_agent.py`: 流量探针
    *   `kdd_mapper.py`: 特征工程映射
*   `ckpt/`: 存放模型权重文件
*   `admin_train/`: 训练脚本与原始数据
