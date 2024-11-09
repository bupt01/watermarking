import os
import time
import psutil
import subprocess
import signal

# 配置部分
process_pid = 1900944  # 需要监控的进程PID
gpu_memory_threshold = 22.2  # 显卡内存阈值，单位为MB
check_interval = 5  # 检查的时间间隔，单位为秒
gpu_id = 1  # 监控的显卡编号

def get_gpu_memory_usage(gpu_id):
    """获取指定显卡的内存使用情况"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        stdout=subprocess.PIPE, text=True
    )
    try:
        gpu_memory_usage = int(result.stdout.strip())
    except ValueError:
        gpu_memory_usage = 0  # 如果获取失败，默认为0
    return gpu_memory_usage




def monitor_and_kill():
    while True:
        # 获取指定显卡的内存使用情况
        gpu_memory_usage = get_gpu_memory_usage(gpu_id)
        print(f"显卡{gpu_id}的当前内存使用: {gpu_memory_usage/1024} GB")

        if gpu_memory_usage/1024 > gpu_memory_threshold:
            try:
                proc = psutil.Process(process_pid)
                print(f"显卡{gpu_id}内存超过阈值，终止进程 PID: {process_pid}")
                proc.kill()
                return
               
            except psutil.NoSuchProcess:
                print(f"进程 PID {process_pid} 不存在，可能已被终止")
                return
           
  
        # 等待一段时间后再检查
        time.sleep(check_interval)
if __name__ == "__main__":
    monitor_and_kill()
