#!/usr/bin/env python3

import subprocess
import json
import csv
import os
import time
import argparse
import re
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceConstraintExperiment:
    def __init__(self, docker_image, config_file):
        self.docker_image = docker_image
        self.results = []
        self.data_path = Path("data")
        self.config_file = config_file

    def load_configurations(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件 {self.config_file} 不存在")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def get_image_files(self):
        """获取数据目录下的图片文件"""
        if not self.data_path.exists():
            logger.error(f"数据目录 {self.data_path} 不存在")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_path.glob(f"*{ext}"))
            image_files.extend(self.data_path.glob(f"*{ext.upper()}"))
        
        return [str(f) for f in image_files]
    
    def parse_memory_limit(self, memory_str):
        """解析内存限制字符串，转换为Docker支持的格式"""
        if memory_str.endswith('Gi'):
            return memory_str.replace('Gi', 'g')
        elif memory_str.endswith('GB'):
            return memory_str.lower()
        elif memory_str.endswith('Mi'):
            return memory_str.replace('Mi', 'm')
        elif memory_str.endswith('MB'):
            return memory_str.lower().replace('mb', 'm')
        else:
            return memory_str
    
    def build_docker_command(self, config, task, image_path):
        """构建Docker运行命令"""
        resources = config["resources"]
        
        # 基础Docker命令
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",  # 启用GPU支持
            "-e", "LD_PRELOAD=/libvgpu/build/libvgpu.so",  # HAMi-Core vGPU支持
            "-e", "HF_HOME=/huggingface_cache",
            "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/huggingface_cache",
        ]
        
        # CPU限制
        if "cpu" in resources:
            cmd.extend(["--cpus", str(resources["cpu"])])
        
        # 内存限制
        if "memory" in resources:
            memory_limit = self.parse_memory_limit(resources["memory"])
            cmd.extend(["-m", memory_limit])
            # 设置交换内存与内存限制相同
            cmd.extend(["--memory-swap", "8g"])
        
        # GPU SM限制 (HAMi-Core)
        if "gpu" in resources:
            gpu_sm = resources["gpu"]
            cmd.extend(["-e", f"CUDA_DEVICE_SM_LIMIT={gpu_sm}"])
        
        # GPU内存限制 (HAMi-Core)
        if "gpumem" in resources:
            cmd.extend(["-e", f"CUDA_DEVICE_MEMORY_LIMIT={resources['gpumem']}m"])
        
        # 挂载数据目录
        current_dir = os.getcwd()
        cmd.extend(["-v", f"{current_dir}/data:/app/data",
                    "-v", f"{current_dir}/inference.py:/app/inference.py",
                    "-v", f"{current_dir}/vgpulock:/tmp/vgpulock"
                ])

        # Docker镜像
        cmd.append(self.docker_image)
        
        # Python执行命令
        cmd.extend([
            "python", "inference.py",
            "--task", task,
            "--device", "cuda",
            # "--image_path", f"/app/{image_path}",
            # "--gpu_memory", resources.get('gpumem', "8g")
        ])
        
        return cmd
    
    def extract_latency_from_output(self, output):
        """从输出中提取延迟信息"""
        latency_info = {}
        
        # 提取end to end时间
        end_to_end_pattern = r"(\w+) end to end time = ([\d.]+)"
        match = re.search(end_to_end_pattern, output)
        if match:
            latency_info['end_to_end_time'] = float(match.group(2))
        
        # 提取内存使用信息
        gpu_current_pattern = r"GPU Current Memory Allocated:\s+([\d.]+) GB"
        gpu_peak_pattern = r"GPU Peak Memory Allocated:\s+([\d.]+) GB"
        cpu_memory_pattern = r"CPU Process Memory \(RSS\):\s+([\d.]+) GB"
        
        gpu_current_match = re.search(gpu_current_pattern, output)
        gpu_peak_match = re.search(gpu_peak_pattern, output)
        cpu_memory_match = re.search(cpu_memory_pattern, output)
        
        if gpu_current_match:
            latency_info['gpu_current_memory'] = float(gpu_current_match.group(1))
        if gpu_peak_match:
            latency_info['gpu_peak_memory'] = float(gpu_peak_match.group(1))
        if cpu_memory_match:
            latency_info['cpu_memory'] = float(cpu_memory_match.group(1))
        
        return latency_info
    
    def run_single_experiment(self, config, task, image_path, repeat=3):
        """运行单个实验配置"""
        logger.info(f"运行实验: {config['name']} - {task} - {os.path.basename(image_path)}")
        
        results = []
        
        for i in range(repeat):
            logger.info(f"  第 {i+1}/{repeat} 次执行")
            
            cmd = self.build_docker_command(config, task, image_path)
            logger.debug(f"执行命令: {' '.join(cmd)}")
            
            try:
                start_time = time.time()
                
                # 使用Popen来实时显示输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时读取并显示输出
                output_lines = []
                logger.info(f"    开始执行Docker命令...")
                
                try:
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            output_line = output.strip()
                            print(f"    [Docker] {output_line}")  # 实时打印输出
                            output_lines.append(output_line)
                    
                    # 等待进程完成
                    process.wait(timeout=300)
                    
                except subprocess.TimeoutExpired:
                    logger.warning("    Docker进程执行超时，正在终止...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                
                end_time = time.time()
                
                # 合并所有输出
                full_output = '\n'.join(output_lines)
                
                # 创建类似subprocess.run的结果对象
                class MockResult:
                    def __init__(self, returncode, stdout, stderr=""):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr
                
                result = MockResult(process.returncode, full_output)
                
                if result.returncode == 0:
                    # 解析输出获取延迟信息
                    # latency_info = self.extract_latency_from_output(result.stdout)
                    
                    experiment_result = {
                        'timestamp': datetime.now().isoformat(),
                        'config_name': config['name'],
                        'task': task,
                        'image_file': os.path.basename(image_path),
                        'repeat_index': i + 1,
                        'cpu_cores': config['resources'].get('cpu', 'N/A'),
                        'memory_limit': config['resources'].get('memory', 'N/A'),
                        'gpu_sm_limit': config['resources'].get('gpu', 'N/A'),
                        'gpu_memory_limit': config['resources'].get('gpumem', 'N/A'),
                        'docker_execution_time': end_time - start_time,
                        # **latency_info,
                        'success': True,
                        'error_message': None
                    }
                    
                    results.append(experiment_result)
                    # logger.info(f"    执行成功, 延迟: {latency_info.get('end_to_end_time', 'N/A'):.2f}s")
                    
                else:
                    error_result = {
                        'timestamp': datetime.now().isoformat(),
                        'config_name': config['name'],
                        'task': task,
                        'image_file': os.path.basename(image_path),
                        'repeat_index': i + 1,
                        'cpu_cores': config['resources'].get('cpu', 'N/A'),
                        'memory_limit': config['resources'].get('memory', 'N/A'),
                        'gpu_sm_limit': config['resources'].get('gpu', 'N/A'),
                        'gpu_memory_limit': config['resources'].get('gpumem', 'N/A'),
                        'docker_execution_time': end_time - start_time,
                        'success': False,
                        'error_message': full_output  # 现在错误信息也在stdout中
                    }
                    
                    results.append(error_result)
                    logger.error(f"    执行失败，输出: {full_output[-500:] if len(full_output) > 500 else full_output}")  # 只显示最后500字符
                    
            except subprocess.TimeoutExpired:
                end_time = time.time()
                error_result = {
                    'timestamp': datetime.now().isoformat(),
                    'config_name': config['name'],
                    'task': task,
                    'image_file': os.path.basename(image_path),
                    'repeat_index': i + 1,
                    'cpu_cores': config['resources'].get('cpu', 'N/A'),
                    'memory_limit': config['resources'].get('memory', 'N/A'),
                    'gpu_sm_limit': config['resources'].get('gpu', 'N/A'),
                    'gpu_memory_limit': config['resources'].get('gpumem', 'N/A'),
                    'docker_execution_time': end_time - start_time,
                    'success': False,
                    'error_message': 'Timeout after 300 seconds'
                }
                
                results.append(error_result)
                logger.error(f"    执行超时")
                
            except Exception as e:
                error_result = {
                    'timestamp': datetime.now().isoformat(),
                    'config_name': config['name'],
                    'task': task,
                    'image_file': os.path.basename(image_path),
                    'repeat_index': i + 1,
                    'cpu_cores': config['resources'].get('cpu', 'N/A'),
                    'memory_limit': config['resources'].get('memory', 'N/A'),
                    'gpu_sm_limit': config['resources'].get('gpu', 'N/A'),
                    'gpu_memory_limit': config['resources'].get('gpumem', 'N/A'),
                    'success': False,
                    'error_message': str(e)
                }
                
                results.append(error_result)
                logger.error(f"    执行异常: {e}")
        
        return results
    
    def save_results_to_csv(self, results, output_file):
        """保存结果到CSV文件"""
        if not results:
            logger.warning("没有结果需要保存")
            return
        
        # 获取所有可能的字段名
        fieldnames = set()
        for result in results:
            fieldnames.update(result.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        # 写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"结果已保存到 {output_file}")
    
    def run_experiments(self, config_filter=None, task_filter=None, image_filter=None, repeat=3):
        """运行完整的实验流程"""
        logger.info("start resource constraint experiments")
        
        # 加载配置
        configurations = self.load_configurations()
        configs = configurations['configurations']
        
        # 获取图片文件
        image_files = self.get_image_files()
        if not image_files:
            logger.error("no image files found")
            return
        
        # 过滤配置
        if config_filter:
            configs = [c for c in configs if c['name'] in config_filter]
        
        # 过滤图片
        if image_filter:
            image_files = [f for f in image_files if any(pattern in f for pattern in image_filter)]
        
        logger.info(f"找到 {len(configs)} 个配置, {len(image_files)} 个图片文件")
        
        all_results = []
        
        for config in configs:
            for task in config['tasks']:
                # 过滤任务
                if task_filter and task not in task_filter:
                    continue
                
                for image_path in image_files:
                    results = self.run_single_experiment(config, task, image_path, repeat)
                    
                    all_results.extend(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{timestamp}.csv"
        self.save_results_to_csv(all_results, output_file)
        
        # 打印统计信息
        successful_experiments = sum(1 for r in all_results if r.get('success', False))
        total_experiments = len(all_results)
        
        logger.info(f"实验完成! 成功: {successful_experiments}/{total_experiments}")
        logger.info(f"结果文件: {output_file}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="资源约束实验执行脚本")
    parser.add_argument("--config", nargs="+", help="指定要运行的配置名称")
    parser.add_argument("--config-file", default="generated_resource_configurations.json", help="配置文件路径")
    parser.add_argument("--task", nargs="+", help="指定要运行的任务名称")
    parser.add_argument("--image", nargs="+", help="指定要使用的图片文件模式")
    parser.add_argument("--repeat", type=int, default=1, help="每个实验重复次数")
    parser.add_argument("--docker-image", default="model-runner", help="Docker镜像名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    experiment = ResourceConstraintExperiment(docker_image=args.docker_image, config_file=args.config_file)
    
    try:
        results = experiment.run_experiments(
            config_filter=args.config,
            task_filter=args.task,
            image_filter=args.image,
            repeat=args.repeat
        )
        
        print(f"\n实验完成! 共执行了 {len(results)} 个实验")
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验执行失败: {e}")
        raise


if __name__ == "__main__":
    main()