import torch
import psutil

def check_memory(model, device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
    
    print(f"GPU Memory Allocated: {allocated_memory:.2f} GB")
    print(f"GPU Memory Reserved: {reserved_memory:.2f} GB")

def check_cpu_memory():
    memory_info = psutil.virtual_memory()

    total_memory = memory_info.total / (1024 ** 3)  
    available_memory = memory_info.available / (1024 ** 3)
    used_memory = memory_info.used / (1024 ** 3)  
    memory_percent = memory_info.percent 

    print(f"Total CPU Memory: {total_memory:.2f} GB")
    print(f"Available CPU Memory: {available_memory:.2f} GB")
    print(f"Used CPU Memory: {used_memory:.2f} GB")
    print(f"CPU Memory Usage: {memory_percent}%")


