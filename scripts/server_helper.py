#!/usr/bin/env /usr/bin/python3
"""
服务器操作辅助脚本
通过 paramiko SSH 连接服务器，提供 GPU 检查、远程命令执行、文件增量同步等能力。

用法:
  python3 scripts/server_helper.py --server 172.20.60.10 gpu              # 检查GPU
  python3 scripts/server_helper.py --server 172.20.60.10 exec "nvidia-smi" # 执行命令
  python3 scripts/server_helper.py --server 172.20.60.10 upload /local/dir /remote/dir
  python3 scripts/server_helper.py --server 172.20.60.10 download /remote/dir /local/dir
  python3 scripts/server_helper.py --server 172.20.60.10 log /path/to/nohup.out
"""

import os
import sys
import argparse
import yaml
import paramiko
from pathlib import Path
from datetime import datetime
from stat import S_ISDIR, S_ISREG


# ---------- config ----------

def load_config():
    """加载服务器配置文件"""
    config_path = Path(__file__).resolve().parent.parent / '.server_config.yaml'
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print("请先创建 .server_config.yaml（参考 .server_config.yaml.example）")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_server(config, server_key):
    """获取指定服务器配置"""
    servers = config.get('servers', {})
    if server_key not in servers:
        available = ', '.join(servers.keys())
        print(f"错误: 未找到服务器 '{server_key}'，可用: {available}")
        sys.exit(1)
    return servers[server_key]


def create_ssh_client(srv):
    """创建 SSH 客户端并连接"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=srv['host'],
        port=srv.get('port', 22),
        username=srv['user'],
        password=srv['password'],
        timeout=15
    )
    return ssh


# ---------- GPU ----------

def cmd_gpu(srv):
    """检查 GPU 状态"""
    ssh = create_ssh_client(srv)
    try:
        # 先查数量
        _, stdout, stderr = ssh.exec_command(
            'nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null'
        )
        count_out = stdout.read().decode('utf-8').strip()
        err = stderr.read().decode('utf-8').strip()

        if err or not count_out:
            print(f"✗ 无法获取 GPU 信息: {err or '无输出'}")
            return

        gpu_count = int(count_out.split('\n')[0])

        # 查详细信息
        _, stdout, _ = ssh.exec_command(
            'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu '
            '--format=csv,noheader,nounits 2>/dev/null'
        )
        output = stdout.read().decode('utf-8').strip()

        print(f"服务器 {srv['host']} GPU 状态 ({datetime.now().strftime('%H:%M:%S')}):")
        print(f"{'ID':>3} {'名称':<20} {'显存(MB)':>12} {'利用率':>8} {'温度':>6} {'空闲?':>6}")
        print("-" * 70)

        free_gpus = []
        for line in output.split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                idx, name, mem_used, mem_total, util = parts[0], parts[1], parts[2], parts[3], parts[4]
                temp = parts[5] if len(parts) >= 6 else '?'
                mem_pct = (int(mem_used) / int(mem_total)) * 100 if int(mem_total) > 0 else 0
                is_free = mem_pct < 15 and int(util) < 15
                status = "✓ 空闲" if is_free else "  占用"
                print(f"{idx:>3} {name:<20} {mem_used:>5}/{mem_total:<5} {util:>5}% {temp:>4}°C {status}")
                if is_free:
                    free_gpus.append(idx)

        if free_gpus:
            print(f"\n空闲 GPU: {', '.join(free_gpus)}")
            print(f"可用 --device 参数: {','.join(free_gpus)}")
        else:
            print("\n⚠ 所有 GPU 均在使用中")
    finally:
        ssh.close()


# ---------- exec ----------

def cmd_exec(srv, command, workdir=None):
    """在服务器上执行命令并打印输出"""
    ssh = create_ssh_client(srv)
    try:
        full_cmd = command
        if workdir:
            full_cmd = f"cd {workdir} && {command}"
        _, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
        # 实时读取
        for line in iter(stdout.readline, ''):
            if line:
                print(line, end='')
        err = stderr.read().decode('utf-8')
        if err:
            print(err, end='', file=sys.stderr)
        # 等待退出
        exit_code = stdout.channel.recv_exit_status()
        return exit_code
    finally:
        ssh.close()


def cmd_nohup_train(srv, command, workdir, logfile):
    """在服务器上用 nohup 后台启动训练"""
    ssh = create_ssh_client(srv)
    try:
        # 如果服务器配置了 conda_env，自动加 conda activate 前缀
        conda_env = srv.get('conda_env', '').strip()
        if conda_env:
            # source conda.sh 使 conda activate 在非交互式 SSH 中可用
            command = f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {command}"
        
        full_cmd = f"cd {workdir} && nohup {command} > {logfile} 2>&1 &"
        print(f"[{srv['host']}] 启动训练:")
        print(f"  conda环境: {conda_env or '(无)'}")
        print(f"  工作目录: {workdir}")
        print(f"  日志文件: {logfile}")
        
        channel = ssh.get_transport().open_session()
        channel.exec_command(full_cmd)
        import time
        time.sleep(2)

        _, check_stdout, _ = ssh.exec_command(
            "ps aux | grep -v grep | grep 'python.*train' | awk '{print $2, $11, $12, $13, $14}'"
        )
        process_info = check_stdout.read().decode('utf-8').strip()
        if process_info:
            print(f"  进程已启动: {process_info}")
        else:
            print(f"  训练命令已提交，检查日志确认: {logfile}")
        print(f"\n后续查看状态: python3 scripts/server_helper.py --server {srv['host']} log {logfile}")
    finally:
        ssh.close()


def cmd_log(srv, logfile, tail=50):
    """查看远程日志文件"""
    ssh = create_ssh_client(srv)
    try:
        _, stdout, _ = ssh.exec_command(
            f"tail -{tail} {logfile} 2>/dev/null || echo '[文件不存在或为空]'"
        )
        print(stdout.read().decode('utf-8'))
    finally:
        ssh.close()


# ---------- file sync ----------

def _get_local_files(local_dir):
    """获取本地目录下所有文件的 (相对路径, 大小, mtime)"""
    files = {}
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return files
    for f in local_dir.rglob('*'):
        if f.is_file():
            rel = str(f.relative_to(local_dir))
            files[rel] = {'size': f.stat().st_size, 'mtime': f.stat().st_mtime}
    return files


def _get_remote_files(sftp, remote_dir):
    """获取远程目录下所有文件的 (相对路径, 大小, mtime)"""
    files = {}
    try:
        sftp.chdir(remote_dir)
    except IOError:
        return files

    def walk_remote(path):
        try:
            for entry in sftp.listdir_attr(path):
                full = f"{path}/{entry.filename}" if path != '.' else entry.filename
                if S_ISDIR(entry.st_mode):
                    walk_remote(full)
                elif S_ISREG(entry.st_mode):
                    files[full] = {'size': entry.st_size, 'mtime': entry.st_mtime}
        except IOError:
            pass

    walk_remote('.')
    return files


def cmd_upload(srv, local_dir, remote_dir, dry_run=False):
    """增量上传：本地 → 服务器"""
    local_dir = str(Path(local_dir).resolve())
    local_files = _get_local_files(local_dir)

    ssh = create_ssh_client(srv)
    sftp = ssh.open_sftp()
    try:
        # 确保远程目录存在
        _mkdir_p(sftp, remote_dir)

        remote_files = _get_remote_files(sftp, remote_dir)
        total = len(local_files)
        upload_count = 0
        skip_count = 0
        total_bytes = 0

        print(f"同步: {local_dir} → {srv['host']}:{remote_dir}")
        print(f"本地文件数: {total}")

        for rel_path, info in sorted(local_files.items()):
            remote_key = rel_path
            if remote_key in remote_files:
                r = remote_files[remote_key]
                # 比较大小和 mtime（允许 2 秒误差）
                if r['size'] == info['size'] and abs(r['mtime'] - info['mtime']) <= 2:
                    skip_count += 1
                    continue

            total_bytes += info['size']
            upload_count += 1

            if dry_run:
                print(f"  [DRY RUN] 将上传: {rel_path} ({_human_size(info['size'])})")
            else:
                local_path = os.path.join(local_dir, rel_path)
                remote_path = f"{remote_dir.rstrip('/')}/{rel_path}"
                _mkdir_p(sftp, str(Path(remote_path).parent))
                sftp.put(local_path, remote_path)
                if upload_count % 100 == 0:
                    print(f"  已上传 {upload_count} 个文件...")

        print(f"完成: 上传 {upload_count} 个, 跳过 {skip_count} 个, "
              f"总计 {_human_size(total_bytes)}")
        if dry_run and upload_count > 0:
            print("(dry-run 模式，未实际传输)")
    finally:
        sftp.close()
        ssh.close()


def cmd_download(srv, remote_dir, local_dir, dry_run=False):
    """增量下载：服务器 → 本地"""
    local_dir = str(Path(local_dir).resolve())
    os.makedirs(local_dir, exist_ok=True)

    ssh = create_ssh_client(srv)
    sftp = ssh.open_sftp()
    try:
        remote_files = _get_remote_files(sftp, remote_dir)
        local_files = _get_local_files(local_dir)

        total = len(remote_files)
        download_count = 0
        skip_count = 0
        total_bytes = 0

        print(f"同步: {srv['host']}:{remote_dir} → {local_dir}")
        print(f"远程文件数: {total}")

        for rel_path, info in sorted(remote_files.items()):
            if rel_path in local_files:
                l = local_files[rel_path]
                if l['size'] == info['size'] and abs(l['mtime'] - info['mtime']) <= 2:
                    skip_count += 1
                    continue

            total_bytes += info['size']
            download_count += 1

            if dry_run:
                print(f"  [DRY RUN] 将下载: {rel_path} ({_human_size(info['size'])})")
            else:
                local_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                remote_path = f"{remote_dir.rstrip('/')}/{rel_path}"
                sftp.get(remote_path, local_path)
                if download_count % 100 == 0:
                    print(f"  已下载 {download_count} 个文件...")

        print(f"完成: 下载 {download_count} 个, 跳过 {skip_count} 个, "
              f"总计 {_human_size(total_bytes)}")
        if dry_run and download_count > 0:
            print("(dry-run 模式，未实际传输)")
    finally:
        sftp.close()
        ssh.close()


# ---------- utils ----------

def _mkdir_p(sftp, remote_dir):
    """递归创建远程目录"""
    dirs = []
    path = remote_dir.rstrip('/')
    while path and path != '/':
        try:
            sftp.stat(path)
            break
        except IOError:
            dirs.append(path)
            path = str(Path(path).parent)
    for d in reversed(dirs):
        sftp.mkdir(d)


def _human_size(size):
    """人类可读的文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description='服务器操作辅助工具')
    parser.add_argument('--server', '-s', type=str, required=True,
                        help='服务器标识（如 172.20.60.10）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示将要同步的文件，不实际传输')

    sub = parser.add_subparsers(dest='action', required=True)

    # gpu
    sub.add_parser('gpu', help='检查 GPU 状态')

    # exec
    p_exec = sub.add_parser('exec', help='执行远程命令')
    p_exec.add_argument('command', type=str, help='要执行的命令')
    p_exec.add_argument('--workdir', '-w', type=str, default=None,
                        help='工作目录')

    # nohup-train
    p_train = sub.add_parser('train', help='后台启动训练（nohup）')
    p_train.add_argument('command', type=str, help='训练命令')
    p_train.add_argument('--workdir', '-w', type=str, required=True,
                         help='服务器工作目录')
    p_train.add_argument('--logfile', '-l', type=str, required=True,
                         help='训练日志文件路径')

    # log
    p_log = sub.add_parser('log', help='查看远程日志')
    p_log.add_argument('logfile', type=str, help='日志文件路径')
    p_log.add_argument('--tail', '-n', type=int, default=50,
                       help='显示最后 N 行（默认 50）')

    # upload
    p_up = sub.add_parser('upload', help='增量上传本地目录到服务器')
    p_up.add_argument('local_dir', type=str, help='本地目录')
    p_up.add_argument('remote_dir', type=str, help='远程目录')

    # download
    p_down = sub.add_parser('download', help='增量下载服务器目录到本地')
    p_down.add_argument('remote_dir', type=str, help='远程目录')
    p_down.add_argument('local_dir', type=str, help='本地目录')

    args = parser.parse_args()
    config = load_config()
    srv = get_server(config, args.server)

    if args.action == 'gpu':
        cmd_gpu(srv)
    elif args.action == 'exec':
        code = cmd_exec(srv, args.command, args.workdir)
        sys.exit(code)
    elif args.action == 'train':
        cmd_nohup_train(srv, args.command, args.workdir, args.logfile)
    elif args.action == 'log':
        cmd_log(srv, args.logfile, args.tail)
    elif args.action == 'upload':
        cmd_upload(srv, args.local_dir, args.remote_dir, args.dry_run)
    elif args.action == 'download':
        cmd_download(srv, args.remote_dir, args.local_dir, args.dry_run)


if __name__ == '__main__':
    main()
