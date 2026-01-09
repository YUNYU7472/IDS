import requests
import random
import time
import sys

# 目标配置
TARGET_URL = "http://localhost:8080/login"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IDS_Tester/1.0",
    "Content-Type": "application/x-www-form-urlencoded"
}

# ===============================
# 1. 正常流量样本
# ===============================
BENIGN_PAYLOADS = [
    {"user": "admin", "pass": "123456"},
    {"user": "zhangsan", "pass": "password"},
    {"user": "test_user", "pass": "qwer!@#$"},
    {"user": "guest", "pass": "guest123"},
]

# ===============================
# 2. 攻击流量样本 (用于触发检测)
# ===============================
ATTACK_PAYLOADS = {
    "SQL_Injection": [
        {"user": "admin", "pass": "' OR '1'='1"},
        {"user": "admin' --", "pass": "123"},
        {"user": "admin", "pass": "' UNION SELECT 1, database(), user() --"},
    ],
    "XSS_Probe": [
        {"user": "<script>alert('IDS')</script>", "pass": "123"},
        {"user": "admin", "pass": "\"><img src=x onerror=alert(1)>"},
    ],
    "Command_Injection": [
        {"user": "admin; cat /etc/passwd", "pass": "123"},
        {"user": "admin && whoami", "pass": "123"},
    ],
    "Path_Traversal": [
        {"user": "../../../etc/shadow", "pass": "123"},
    ]
}

def send_request(payload, tag="NORMAL"):
    """发送请求并打印状态"""
    try:
        # 蜜罐通常是低交互的，可能返回 200 或 403，这里只负责发
        response = requests.post(TARGET_URL, data=payload, headers=HEADERS, timeout=5)
        print(f"[{time.strftime('%H:%M:%S')}] [{tag}] 发送 Payload: {str(payload)[:60]}... -> 状态码: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 无法连接到蜜罐 ({TARGET_URL})。请确认 honeypot.py 是否运行。")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] 发送失败: {e}")

def run_mixed_test(count=10):
    """混合模式：随机发送正常和攻击流量"""
    print(f"🚀 开始混合流量测试 (共 {count} 次请求)...")
    
    for i in range(count):
        # 30% 概率发送攻击，70% 概率发送正常
        if random.random() < 0.3:
            attack_type = random.choice(list(ATTACK_PAYLOADS.keys()))
            payload = random.choice(ATTACK_PAYLOADS[attack_type])
            send_request(payload, tag=f"ATTACK-{attack_type}")
        else:
            payload = random.choice(BENIGN_PAYLOADS)
            send_request(payload, tag="NORMAL")
        
        # 随机延时，模拟真实间隔
        time.sleep(random.uniform(0.5, 1.5))

def run_specific_attack(attack_type):
    """针对性测试"""
    if attack_type not in ATTACK_PAYLOADS:
        print("未知的攻击类型")
        return

    print(f"🔥 开始针对性测试: {attack_type} ...")
    for payload in ATTACK_PAYLOADS[attack_type]:
        send_request(payload, tag=f"ATTACK-{attack_type}")
        time.sleep(1)

if __name__ == "__main__":
    print("IDS/蜜罐 流量生成器 v1.0")
    print("1. 发送混合流量 (模拟真实环境)")
    print("2. 仅测试 SQL 注入")
    print("3. 仅测试 XSS 跨站脚本")
    print("4. 仅测试 命令注入")
    
    choice = input("请选择测试模式 (1-4): ").strip()
    
    if choice == "1":
        count = input("请输入请求数量 (默认10): ").strip()
        run_mixed_test(int(count) if count.isdigit() else 10)
    elif choice == "2":
        run_specific_attack("SQL_Injection")
    elif choice == "3":
        run_specific_attack("XSS_Probe")
    elif choice == "4":
        run_specific_attack("Command_Injection")
    else:
        print("无效选择，退出。")