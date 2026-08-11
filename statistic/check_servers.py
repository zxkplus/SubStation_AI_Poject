#!/usr/bin/env python3
"""
服务器GPU状态检查脚本
从配置文件读取服务器信息，连接各服务器并检查GPU使用情况。
"""

import paramiko
import csv
import os
import sys
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_config.csv")

def read_server_info(config_path=CONFIG_PATH):
    """从CSV配置文件读取服务器信息"""
    server_info = []

    if not os.path.exists(config_path):
        print(f"错误: 服务器配置文件不存在: {config_path}")
        print("请复制 statistic/server_config.example.csv 为 server_config.csv 并填入服务器信息。")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 跳过空行和标题行
    for row in rows:
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].strip().lower() in ('ip', 'host', '#ip'):
            continue
        if len(row) < 3:
            print(f"警告: 跳过格式异常的行: {row}")
            continue
        server_info.append({
            'ip': row[0].strip(),
            'username': row[1].strip(),
            'password': row[2].strip()
        })

    return server_info

def check_gpu_status(hostname, username, password, port=22):
    """连接服务器并检查GPU状态"""
    try:
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接服务器
        ssh.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        # 执行nvidia-smi命令获取GPU数量
        stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=count --format=csv,noheader,nounits')
        count_output = stdout.read().decode('utf-8').strip()
        count_error = stderr.read().decode('utf-8').strip()
        
        if count_error:
            # 如果无法获取数量，尝试直接获取详细信息
            stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits')
            output = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8').strip()
            
            ssh.close()
            
            if error:
                return {"error": f"Error: {error}"}
            
            if not output:
                return {"error": "No GPU information available"}
                
            # 解析GPU信息
            gpu_lines = output.split('\n')
            gpus = []
            for gpu_line in gpu_lines:
                if gpu_line.strip():
                    parts = gpu_line.split(',')
                    if len(parts) >= 5:
                        gpus.append({
                            'index': parts[0].strip(),
                            'name': parts[1].strip(),
                            'memory_used': parts[2].strip(),
                            'memory_total': parts[3].strip(),
                            'utilization': parts[4].strip()
                        })
            
            return {
                'gpu_count': len(gpus),
                'gpus': gpus
            }
        else:
            # 获取GPU详细信息
            stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits')
            output = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8').strip()
            
            ssh.close()
            
            if error:
                return {"error": f"Error: {error}"}
            
            if not output:
                return {"error": "No GPU information available"}
                
            # 解析GPU信息
            gpu_lines = output.split('\n')
            gpus = []
            for gpu_line in gpu_lines:
                if gpu_line.strip():
                    parts = gpu_line.split(',')
                    if len(parts) >= 5:
                        gpus.append({
                            'index': parts[0].strip(),
                            'name': parts[1].strip(),
                            'memory_used': parts[2].strip(),
                            'memory_total': parts[3].strip(),
                            'utilization': parts[4].strip()
                        })
            
            return {
                'gpu_count': len(gpus),
                'gpus': gpus
            }
        
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

def main():
    """主函数"""
    print(f"开始检查服务器GPU状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    servers = read_server_info()
    
    for i, server in enumerate(servers):
        if not server['ip']:
            continue
            
        print(f"\n服务器 #{i+1}: {server['ip']}")
        print(f"用户: {server['username']}")
        print("-" * 40)
        
        gpu_info = check_gpu_status(server['ip'], server['username'], server['password'])
        
        if "error" in gpu_info:
            print(f"✗ {gpu_info['error']}")
        else:
            print(f"✓ 显卡数量: {gpu_info['gpu_count']}")
            if gpu_info['gpu_count'] > 0:
                print("显卡占用情况:")
                for gpu in gpu_info['gpus']:
                    memory_percent = (int(gpu['memory_used']) / int(gpu['memory_total'])) * 100 if int(gpu['memory_total']) > 0 else 0
                    print(f"  GPU {gpu['index']}: {gpu['name']}")
                    print(f"    显存使用: {gpu['memory_used']}/{gpu['memory_total']} MB ({memory_percent:.1f}%)")
                    print(f"    GPU利用率: {gpu['utilization']}%")
            else:
                print("  未检测到GPU设备")
        
        print("=" * 80)

if __name__ == "__main__":
    main()