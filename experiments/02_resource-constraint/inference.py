import torch
import time
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
import os
import functools
import click
import psutil

def timer(func):
    """一个打印函数执行时间的装饰器"""
    @functools.wraps(func)  # 保持被装饰函数原有的元信息（如__name__, __doc__）
    def wrapper(*args, **kwargs):
        # 1. 记录开始时间
        start_time = time.perf_counter()

        # 2. 执行原始函数，并获取其返回值
        value = func(*args, **kwargs)

        # 3. 记录结束时间并计算差值
        end_time = time.perf_counter()
        run_time = end_time - start_time

        # 4. 打印执行时间
        # 使用 !r 来获取函数名的官方表示（例如 '<function my_function at 0x...>'）
        print(f"函数 {func.__name__!r} 执行完毕，耗时 {run_time:.4f} 秒")

        # 5. 返回原始函数的执行结果
        return value
    return wrapper

def super_resolution(max_memory_mapping, image, device):
    model_name = "caidas/swin2SR-classical-sr-x2-64"

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        # device_map="auto",
        # max_memory=max_memory_mapping,
        local_files_only=True,

    )

    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_name,
        # device_map="auto",
        # max_memory=max_memory_mapping,
        local_files_only=True,
    ).to(device)

    model = model.to(device)


    time0 = time.time()
    inputs = processor(
        images=image, return_tensors="pt"
    ).to(device)
    time1 = time.time()
    # print(f"image processing time = {(time1 - time0):.2f}")

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    time1 = time.time()
    # print(f"inference time = {(time1 - start_time):.2f}")
    

# outputs.reconstruction is shape: (B, C, H, W)
    sr_images = []
    reconstructions = outputs.reconstruction.clamp_(0, 1)
    for i in range(reconstructions.shape[0]):
        out_img = reconstructions[i]  # .permute(1, 2, 0)
        out_img = (out_img.cpu().numpy() * 255.0).round().astype(np.uint8)
        sr_images.append(torch.from_numpy(out_img))

    new_data = {"image": sr_images}

    end_time = time.time()
    # peak_after_load = check_memory_usage("Super Resolution")
    # print(f"super resolution time = {end_time - start_time :.2f}")

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
    # peak_after_load = check_memory_usage("before load")
    pixel_values = processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)
    # peak_after_load = check_memory_usage("after load")

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values, max_length=40, num_beams=4
        )
    # peak_after_load = check_memory_usage("after compute")
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    end_time = time.time()
    # print(f"image captioning time = {(end_time - start_time):.2f}")
    captions = [p.strip() for p in preds]

    new_data = {"text-caption": captions}

def object_detection(max_memory_mapping, image, device):
    model_name = "facebook/detr-resnet-101"
    device = "cuda"
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

    inputs = processor(
        images=image, return_tensors="pt"
    ).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)

    threshold = 0.9
    target_sizes = torch.tensor([
        [inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]]
        for _ in range(inputs["pixel_values"].shape[0])
    ]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )
    end_time = time.time()
    # print(f"object detection time = {(end_time - start_time):.2f}")
    # check_memory_usage("object detection")

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
            output += label_name
            output += ", "
        predicted_results.append(output[:-2])

        new_data = {"object_detection_information": final_outputs, 'text-object': predicted_results}


def check_memory_usage(label):
    """检测GPU和CPU内存使用情况"""
    # GPU内存检测
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.synchronize()
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3) # 转换为 GB
        current_gpu_memory = torch.cuda.memory_allocated() / (1024**3) # 转换为 GB
    else:
        peak_gpu_memory = 0
        current_gpu_memory = 0

    # CPU内存检测
    process = psutil.Process()
    memory_info = process.memory_info()
    current_cpu_memory = memory_info.rss / (1024**3)  # 转换为 GB (Resident Set Size - 物理内存)

    # 系统总体内存使用情况
    system_memory = psutil.virtual_memory()
    total_system_memory = system_memory.total / (1024**3)  # 总内存 GB
    used_system_memory = system_memory.used / (1024**3)   # 已使用内存 GB
    available_system_memory = system_memory.available / (1024**3)  # 可用内存 GB

    print(f"\n--- {label} Memory Status ---")
    if gpu_available:
        print(f"GPU Current Memory Allocated: {current_gpu_memory:.2f} GB")
        print(f"GPU Peak Memory Allocated:    {peak_gpu_memory:.2f} GB")
    else:
        print("GPU: Not available")

    print(f"CPU Process Memory (RSS):     {current_cpu_memory:.2f} GB")
    print(f"System Total Memory:          {total_system_memory:.2f} GB")
    print(f"System Used Memory:           {used_system_memory:.2f} GB")
    print(f"System Available Memory:      {available_system_memory:.2f} GB")
    print(f"System Memory Usage:          {(used_system_memory/total_system_memory)*100:.1f}%")

    return peak_gpu_memory if gpu_available else 0

@click.command()
@click.option("--gpu_memory", default = "3GB")
@click.option("--task", default="ImageCaptioning")
@click.option("--device", default="cuda")
@click.option("--image_path", default="lena.png")
def main(gpu_memory, task, device, image_path):
    max_memory_mapping = {
        0: gpu_memory,
        "cpu": "30GB"   # 允许 CPU 使用 30GB
    }

    if torch.cuda.is_available():
        device_id = 0 # 假设使用 GPU 0
        torch.cuda.set_device(device_id)
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        print("CUDA not available. GPU usage monitoring will be skipped.")
        exit()

    torch.cuda.empty_cache() # 清空缓存 (可选，但推荐)
    torch.cuda.reset_peak_memory_stats(device_id) # **关键：重置峰值记录**

    image = Image.open(image_path)

# super_resolution(max_memory_mapping)
    time1 = time.time()
    match task:
        case "ImageCaptioning":
            image_captioning(max_memory_mapping, image, device)
        case "SuperResolution":
            super_resolution(max_memory_mapping, image, device)
        case "ObjectDetection":
            object_detection(max_memory_mapping, image, device)
    time2 = time.time()
    # peak_after_load = check_memory_usage("After computing")
    torch.cuda.reset_peak_memory_stats(device_id)

    print(f"{task} end to end time = {(time2 - time1):.2f}")

if __name__ == "__main__":
    main()

