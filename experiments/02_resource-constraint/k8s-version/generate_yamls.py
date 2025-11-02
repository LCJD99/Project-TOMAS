#!/usr/bin/env python3
"""
批量生成Kubernetes YAML配置文件的脚本

此脚本使用Jinja2模板引擎，根据资源配置JSON文件批量生成不同资源配置的Kubernetes Pod YAML文件。
支持多种AI模型任务（ImageCaptioning、SuperResolution、ObjectDetection）和不同的资源配置。
"""

import json
import os
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def load_configurations(config_file):
    """加载资源配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_file} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件格式错误 - {e}")
        return None


def setup_jinja_environment(template_dir):
    """设置Jinja2环境"""
    try:
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        return env
    except Exception as e:
        print(f"错误: 无法设置Jinja2环境 - {e}")
        return None


def generate_yaml_files(configs, template_name, output_dir, generate_all_tasks=False):
    """生成YAML文件"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置Jinja2环境
    template_dir = Path.cwd()
    env = setup_jinja_environment(template_dir)
    if not env:
        return False
    
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"错误: 无法加载模板文件 {template_name} - {e}")
        return False
    
    generated_files = []
    
    for config in configs['configurations']:
        config_name = config['name']
        resources = config['resources']
        tasks = config.get('tasks', [])
        
        print(f"正在处理配置: {config_name}")
        print(f"  资源配置: CPU={resources['cpu']}, Memory={resources['memory']}, GPU={resources['gpu']}, GPUMem={resources['gpumem']}")
        
        if generate_all_tasks:
            # 为每个任务单独生成YAML文件
            for task in tasks:
                task_info = configs['tasks'].get(task, {})
                
                # 准备模板变量
                template_vars = {
                    'pod_name': f"{config_name}-{task.lower()}",
                    'experiment_label': 'resource-constraint-experiment',
                    'task': task.lower(),
                    'task_type': task,
                    'config_name': config_name,
                    'resources': resources,
                    'container_name': f"{config_name}-{task.lower()}-container",
                    'command': ['python', 'inference.py'],
                }
                
                # 渲染模板
                try:
                    yaml_content = template.render(**template_vars)
                    
                    # 生成文件名
                    filename = f"{config_name}_{task.lower()}.yaml"
                    file_path = output_path / filename
                    
                    # 写入文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(yaml_content)
                    
                    generated_files.append(str(file_path))
                    print(f"  ✓ 生成: {filename} (任务: {task})")
                    
                except Exception as e:
                    print(f"  ✗ 生成失败: {config_name}_{task.lower()}.yaml - {e}")
        else:
            # 为每个配置生成一个通用的YAML文件（不指定特定任务）
            template_vars = {
                'pod_name': f"{config_name}-pod",
                'experiment_label': 'resource-constraint-experiment', 
                'task': 'inference',
                'config_name': config_name,
                'resources': resources,
                'container_name': f"{config_name}-container",
                'command': ['python', 'inference.py'],
            }
            
            # 渲染模板
            try:
                yaml_content = template.render(**template_vars)
                
                # 生成文件名
                filename = f"{config_name}.yaml"
                file_path = output_path / filename
                
                # 写入文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
                
                generated_files.append(str(file_path))
                print(f"  ✓ 生成: {filename}")
                
            except Exception as e:
                print(f"  ✗ 生成失败: {config_name}.yaml - {e}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description='批量生成Kubernetes YAML配置文件')
    parser.add_argument('--config', '-c', default='resource_configurations.json',
                       help='资源配置JSON文件路径 (默认: resource_configurations.json)')
    parser.add_argument('--template', '-t', default='template.yaml.j2',
                       help='Jinja2模板文件路径 (默认: template.yaml.j2)')
    parser.add_argument('--output', '-o', default='generated_yamls',
                       help='输出目录 (默认: generated_yamls)')
    parser.add_argument('--all-tasks', action='store_true',
                       help='为每个配置的每个任务单独生成YAML文件')
    
    args = parser.parse_args()
    
    
    # 加载配置
    configs = load_configurations(args.config)
    if not configs:
        return 1
    
    # 生成YAML文件
    generated_files = generate_yaml_files(
        configs, 
        args.template, 
        args.output,
        args.all_tasks
    )
    
    if not generated_files:
        print("no file")
        return 1
    
    print("-" * 50)
    print(f" generate {len(generated_files)} files ")
    return 0


if __name__ == "__main__":
    exit(main())