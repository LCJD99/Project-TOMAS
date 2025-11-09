#!/usr/bin/env python3
import json
import itertools
from typing import Dict, List, Any

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """从配置文件加载配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_resource_configurations(config_file: str = "config.json") -> Dict[str, Any]:
    
    # 从配置文件读取参数
    config_data = load_config(config_file)
    
    resource_options = {
        "cpu": [str(cpu) for cpu in config_data["cpu_core"]],
        "memory": config_data["cpu_memory"],
        "gpu": config_data["gpu_core"],
        "gpumem": config_data["gpu_memory"]
    }
    
    tasks = config_data["task"]

    # 生成所有可能的资源组合
    combinations = list(itertools.product(
        resource_options["cpu"],
        resource_options["memory"],
        resource_options["gpu"],
        resource_options["gpumem"]
    ))

    configurations = []

    # 以task为最外层循环
    for task in tasks:
        for i, (cpu, memory, gpu, gpumem) in enumerate(combinations):
            # 根据资源组合生成配置名称
            config_name = f"{task}-cpu{cpu}-mem{memory}-gpu{gpu}-gpumem{gpumem}"

            # 生成描述
            description = f"Task: {task}, Resource configuration: CPU={cpu}, Memory={memory}, GPU={gpu}%, GPU Memory={gpumem}"

            # 创建配置对象
            config = {
                "name": config_name,
                "description": description,
                "task": task,  # 单独存储任务类型
                "resources": {
                    "cpu": cpu,
                    "memory": memory,
                    "gpu": gpu,
                    "gpumem": gpumem
                },
                "tasks": [task]  # 保持向后兼容性
            }

            configurations.append(config)

    # 定义任务信息
    task_definitions = {}
    for task in tasks:
        if task == "ImageCaptioning":
            task_definitions[task] = {
                "description": "Image captioning task using ViT-GPT2 model",
                "model": "nlpconnect/vit-gpt2-image-captioning"
            }
        elif task == "ObjectDetection":
            task_definitions[task] = {
                "description": "Object detection task",
                "model": "detection-model"
            }
        elif task == "SuperResolution":
            task_definitions[task] = {
                "description": "Super resolution task",
                "model": "super-resolution-model"
            }
        else:
            task_definitions[task] = {
                "description": f"{task} task",
                "model": f"{task.lower()}-model"
            }

    # 构建最终的配置文件结构
    result = {
        "configurations": configurations,
        "tasks": task_definitions
    }

    return result

def main():
    """主函数：生成配置并保存到文件"""

    # 生成配置
    config_data = generate_resource_configurations()

    # 输出文件路径
    output_file = "generated_resource_configurations.json"

    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"generate {len(config_data['configurations'])} configurations")
    
    # 按task统计配置数量
    task_counts = {}
    for config in config_data['configurations']:
        task = config['task']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("Configurations per task:")
    for task, count in task_counts.items():
        print(f"  {task}: {count} configurations")

if __name__ == "__main__":
    main()
