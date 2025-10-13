import torch
import time
import json
import os
import socket
from transformers import (
    AutoImageProcessor,
    Swin2SRForImageSuperResolution,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    DetrImageProcessor,
    DetrForObjectDetection
)
import numpy as np
from PIL import Image
import functools
import click
import psutil

def timer(func):
    """一个打印函数执行时间的装饰器"""
    @functools.wraps(func)  
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"函数 {func.__name__!r} 执行完毕，耗时 {run_time:.4f} 秒")
        return value, run_time
    return wrapper

@timer
def super_resolution(max_memory_mapping, image, device):
    model_name = "caidas/swin2SR-classical-sr-x2-64"

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=True,
    )

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_name,
        local_files_only=True,
    ).to(device)

    time0 = time.time()
    inputs = processor(
        images=image, return_tensors="pt"
    ).to(device)
    time1 = time.time()
    processing_time = time1 - time0

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    # outputs.reconstruction is shape: (B, C, H, W)
    sr_images = []
    reconstructions = outputs.reconstruction.clamp_(0, 1)
    for i in range(reconstructions.shape[0]):
        out_img = reconstructions[i]
        out_img = (out_img.cpu().numpy() * 255.0).round().astype(np.uint8)
        sr_images.append(torch.from_numpy(out_img))

    return {
        "processing_time": processing_time,
        "inference_time": inference_time,
        "output_shape": list(reconstructions.shape)
    }

@timer
def image_captioning(max_memory_mapping, image, device):
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory_mapping,
        local_files_only=True,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory_mapping,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory_mapping,
        local_files_only=True,
    )

    time0 = time.time()
    pixel_values = processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)
    processing_time = time.time() - time0

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values, max_length=40, num_beams=4
        )
    inference_time = time.time() - start_time
    
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [p.strip() for p in preds]

    return {
        "processing_time": processing_time,
        "inference_time": inference_time,
        "captions": captions
    }

@timer
def object_detection(max_memory_mapping, image, device):
    model_name = "facebook/detr-resnet-101"
    processor = DetrImageProcessor.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory_mapping,
        local_files_only=True,
    )
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory_mapping,
        revision="no_timm",
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    time0 = time.time()
    inputs = processor(
        images=image, return_tensors="pt"
    ).to(device)
    processing_time = time.time() - time0

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time

    threshold = 0.9
    target_sizes = torch.tensor([
        [inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]
        for _ in range(inputs["pixel_values"].shape[0])
    ]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )

    final_outputs = []
    predicted_results = []
    for r in results:
        output = ""
        boxes = r["boxes"].cpu().tolist()
        scores = r["scores"].cpu().tolist()
        labels = r["labels"].cpu().tolist()
        label_names = [model.config.id2label[l] for l in labels]
        final_outputs.append(
            {"boxes": boxes, "scores": scores, "labels": label_names}
        )
        for label_name in label_names:
            output += label_name + ", "
        predicted_results.append(output[:-2] if output else "")

    return {
        "processing_time": processing_time,
        "inference_time": inference_time,
        "detections": final_outputs,
        "detected_objects": predicted_results
    }

def get_system_info():
    """获取系统资源信息"""
    # GPU信息
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_cached": torch.cuda.memory_reserved() / (1024**3)
        }
    else:
        gpu_info = {"gpu_available": False}

    # CPU信息
    cpu_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total / (1024**3),
        "memory_available": psutil.virtual_memory().available / (1024**3),
        "memory_used": psutil.virtual_memory().used / (1024**3),
        "memory_percent": psutil.virtual_memory().percent
    }

    # Pod信息
    pod_info = {
        "hostname": socket.gethostname(),
        "pod_name": os.environ.get("HOSTNAME", "unknown"),
        "node_name": os.environ.get("NODE_NAME", "unknown")
    }

    return {
        "gpu_info": gpu_info,
        "cpu_info": cpu_info,
        "pod_info": pod_info
    }

def get_resource_limits():
    """从K8s环境变量或cgroup获取资源限制"""
    limits = {}
    
    # 尝试从cgroup获取CPU限制
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
            cpu_quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
            cpu_period = int(f.read().strip())
        if cpu_quota > 0:
            limits["cpu_cores"] = cpu_quota / cpu_period
    except:
        limits["cpu_cores"] = "unknown"
    
    # 尝试从cgroup获取内存限制
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            memory_limit = int(f.read().strip())
        limits["memory_gb"] = memory_limit / (1024**3)
    except:
        limits["memory_gb"] = "unknown"
    
    # GPU信息从环境变量获取
    limits["gpu_memory"] = os.environ.get("GPU_MEMORY", "unknown")
    
    return limits

@click.command()
@click.option("--gpu_memory", default="3GB")
@click.option("--task", default="ImageCaptioning")
@click.option("--device", default="cuda")
@click.option("--image_path", default="lena.png")
@click.option("--output_file", default="/tmp/experiment_result.json")
def main(gpu_memory, task, device, image_path, output_file):
    max_memory_mapping = {
        0: gpu_memory,
        "cpu": "30GB"
    }

    # 检查CUDA可用性和设备设置
    if torch.cuda.is_available():
        device_id = 0
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        print("CUDA not available. Using CPU.")
        device = "cpu"

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)

    # 获取系统信息
    system_info_before = get_system_info()
    resource_limits = get_resource_limits()
    
    # 加载图像
    image = Image.open(image_path)
    
    # 执行任务
    start_time = time.time()
    task_result = None
    task_execution_time = 0
    
    try:
        if task == "ImageCaptioning":
            task_result, task_execution_time = image_captioning(max_memory_mapping, image, device)
        elif task == "SuperResolution":
            task_result, task_execution_time = super_resolution(max_memory_mapping, image, device)
        elif task == "ObjectDetection":
            task_result, task_execution_time = object_detection(max_memory_mapping, image, device)
        else:
            raise ValueError(f"Unknown task: {task}")
    except Exception as e:
        task_result = {"error": str(e)}
        task_execution_time = 0
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取执行后的系统信息
    system_info_after = get_system_info()
    
    # 构建结果
    result = {
        "experiment_config": {
            "task": task,
            "gpu_memory": gpu_memory,
            "device": device,
            "image_path": image_path
        },
        "resource_limits": resource_limits,
        "system_info_before": system_info_before,
        "system_info_after": system_info_after,
        "task_result": task_result,
        "timing": {
            "total_execution_time": total_time,
            "task_execution_time": task_execution_time
        },
        "timestamp": time.time()
    }
    
    # 输出结果到文件和控制台
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # 简化输出到控制台
    print(f"\n=== EXPERIMENT RESULT ===")
    print(f"Task: {task}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Task Execution Time: {task_execution_time:.4f}s")
    if torch.cuda.is_available():
        print(f"GPU Memory Used: {system_info_after['gpu_info']['gpu_memory_allocated']:.2f}GB")
    print(f"CPU Memory Used: {system_info_after['cpu_info']['memory_percent']:.1f}%")
    print(f"Result saved to: {output_file}")
    
    # 重置GPU内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)

if __name__ == "__main__":
    main()