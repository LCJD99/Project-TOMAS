#!/usr/bin/env python3
import json
import itertools
from typing import Dict, List, Any

def generate_resource_configurations() -> Dict[str, Any]:
    
    resource_options = {
        "cpu": ["1", "2", "4", "16"],
        "memory": ["256m", "512m", "1g", "2g"], 
        "gpu": ["5", "20", "50", "100"],
        "gpumem": ["512m", "800m", "1g", "2g"]
    }
    
    # 生成所有可能的组合
    combinations = list(itertools.product(
        resource_options["cpu"],
        resource_options["memory"], 
        resource_options["gpu"],
        resource_options["gpumem"]
    ))
    
    configurations = []
    
    for i, (cpu, memory, gpu, gpumem) in enumerate(combinations):
        # 根据资源组合生成配置名称
        config_name = f"cpu{cpu}-mem{memory}-gpu{gpu}-gpumem{gpumem}"
        
        # 生成描述
        description = f"Resource configuration: CPU={cpu}, Memory={memory}, GPU={gpu}, GPU Memory={gpumem}MB"
        
        # 创建配置对象
        config = {
            "name": config_name,
            "description": description,
            "resources": {
                "cpu": cpu,
                "memory": memory,
                "gpu": gpu,
                "gpumem": gpumem
            },
            "tasks": ["ImageCaptioning"]
        }
        
        configurations.append(config)
    
    # 定义任务信息
    tasks = {
        "ImageCaptioning": {
            "description": "Image captioning task using ViT-GPT2 model",
            "model": "nlpconnect/vit-gpt2-image-captioning"
        }
    }
    
    # 构建最终的配置文件结构
    result = {
        "configurations": configurations,
        "tasks": tasks
    }
    
    return result

def main():
    """主函数：生成配置并保存到文件"""
    
    print("正在生成256种资源配置组合...")
    
    # 生成配置
    config_data = generate_resource_configurations()
    
    # 输出文件路径
    output_file = "generated_resource_configurations.json"
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"generate {len(config_data['configurations'])} configurations")

if __name__ == "__main__":
    main()