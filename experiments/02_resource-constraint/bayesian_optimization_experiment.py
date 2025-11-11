#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
import warnings
warnings.filterwarnings('ignore')

# 导入现有的实验类
from run_experiment import ResourceConstraintExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bayesian_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BayesianOptimizer:
    def __init__(self, docker_image, seed_data_file, task_filter=None, image_filter=None):
        self.docker_image = docker_image
        self.seed_data_file = seed_data_file
        self.task_filter = task_filter or []
        self.image_filter = image_filter or []
        
        # 加载种子数据
        self.seed_data = self.load_seed_data()
        
        # 定义搜索空间
        self.define_search_space()
        
        # 初始化实验执行器
        self.experiment_executor = ResourceConstraintExperiment(
            docker_image=docker_image, 
            config_file="generated_resource_configurations.json"
        )
        
        # 存储优化历史
        self.optimization_history = []
        
    def load_seed_data(self):
        """加载种子数据"""
        logger.info(f"加载种子数据: {self.seed_data_file}")
        
        try:
            df = pd.read_csv(self.seed_data_file)
            
            # 过滤任务
            if self.task_filter:
                df = df[df['task'].isin(self.task_filter)]
                logger.info(f"过滤任务后剩余 {len(df)} 条数据")
            
            # 过滤图片
            if self.image_filter:
                df = df[df['image_file'].isin(self.image_filter)]
                logger.info(f"过滤图片后剩余 {len(df)} 条数据")
            
            # 只保留成功的实验
            df = df[df['success'] == True]
            logger.info(f"过滤失败实验后剩余 {len(df)} 条数据")
            
            if len(df) == 0:
                raise ValueError("种子数据为空，请检查过滤条件")
                
            logger.info(f"成功加载 {len(df)} 条种子数据")
            return df
            
        except Exception as e:
            logger.error(f"加载种子数据失败: {e}")
            raise
    
    def define_search_space(self):
        """定义搜索空间"""
        # 分析种子数据的范围
        cpu_values = sorted(self.seed_data['cpu_cores'].unique())
        memory_values = sorted(self.seed_data['memory_limit'].unique())
        gpu_memory_values = sorted(self.seed_data['gpu_memory_limit'].unique())
        
        logger.info(f"CPU cores 范围: {cpu_values}")
        logger.info(f"Memory 范围: {memory_values}")
        logger.info(f"GPU memory 范围: {gpu_memory_values}")
        
        # 定义搜索空间
        self.dimensions = [
            Integer(1, 32, name='cpu_cores'),  # CPU核心数
            Categorical(['1g', '2g', '4g', '6g', '8g', '10g', '12g'], name='memory_limit'),  # 内存限制
            Integer(50, 100, name='gpu_sm_limit'),  # GPU SM限制
            Categorical(['800m', '1g', '1.5g', '2g', '3g', '4g', '6g', '8g'], name='gpu_memory_limit')  # GPU内存限制
        ]
        
        # 从种子数据中提取初始点
        self.initial_points = []
        self.initial_values = []
        
        for _, row in self.seed_data.iterrows():
            point = [
                int(row['cpu_cores']),
                row['memory_limit'],
                int(row['gpu_sm_limit']),
                row['gpu_memory_limit']
            ]
            self.initial_points.append(point)
            # 优化目标：最小化执行时间
            self.initial_values.append(row['docker_execution_time'])
        
        logger.info(f"提取了 {len(self.initial_points)} 个初始点")
        
    def convert_params_to_config(self, params):
        """将优化参数转换为实验配置"""
        cpu_cores, memory_limit, gpu_sm_limit, gpu_memory_limit = params
        
        config = {
            'name': f"BayesOpt-cpu{cpu_cores}-mem{memory_limit}-gpu{gpu_sm_limit}-gpumem{gpu_memory_limit}",
            'task': self.task_filter[0] if self.task_filter else 'ImageCaptioning',
            'resources': {
                'cpu': cpu_cores,
                'memory': memory_limit,
                'gpu': gpu_sm_limit,
                'gpumem': gpu_memory_limit
            }
        }
        
        return config
    
    def memory_to_bytes(self, memory_str):
        """将内存字符串转换为字节数"""
        if memory_str.endswith('g'):
            return float(memory_str[:-1]) * 1024**3
        elif memory_str.endswith('m'):
            return float(memory_str[:-1]) * 1024**2
        else:
            return float(memory_str)
    
    def calculate_resource_cost(self, params):
        """计算资源成本（用于多目标优化）"""
        cpu_cores, memory_limit, gpu_sm_limit, gpu_memory_limit = params
        
        # 归一化成本计算
        cpu_cost = cpu_cores / 32  # 最大32核
        memory_cost = self.memory_to_bytes(memory_limit) / self.memory_to_bytes('12g')  # 最大12G
        gpu_sm_cost = gpu_sm_limit / 100  # 最大100
        gpu_memory_cost = self.memory_to_bytes(gpu_memory_limit) / self.memory_to_bytes('8g')  # 最大8G
        
        # 加权资源成本
        total_cost = (cpu_cost * 0.2 + memory_cost * 0.3 + 
                     gpu_sm_cost * 0.3 + gpu_memory_cost * 0.2)
        
        return total_cost
    
    @use_named_args(dimensions=None)  # 将在运行时设置
    def objective_function(self, **params_dict):
        """优化目标函数"""
        # 从字典中提取参数
        params = [params_dict[dim.name] for dim in self.dimensions]
        cpu_cores, memory_limit, gpu_sm_limit, gpu_memory_limit = params
        
        logger.info(f"评估参数: CPU={cpu_cores}, MEM={memory_limit}, GPU_SM={gpu_sm_limit}, GPU_MEM={gpu_memory_limit}")
        
        # 检查是否已经在种子数据中
        existing = self.seed_data[
            (self.seed_data['cpu_cores'] == cpu_cores) &
            (self.seed_data['memory_limit'] == memory_limit) &
            (self.seed_data['gpu_sm_limit'] == gpu_sm_limit) &
            (self.seed_data['gpu_memory_limit'] == gpu_memory_limit)
        ]
        
        if len(existing) > 0:
            execution_time = existing['docker_execution_time'].mean()
            logger.info(f"使用种子数据中的结果: {execution_time:.2f}s")
        else:
            # 运行新实验
            config = self.convert_params_to_config(params)
            execution_time = self.run_experiment(config)
        
        # # 计算复合目标：性能 + 资源成本
        resource_cost = self.calculate_resource_cost(params)
        
        # 目标函数：平衡性能和成本
        # 执行时间权重0.7，资源成本权重0.3
        objective_value = execution_time 
        
        # 记录优化历史
        self.optimization_history.append({
            'cpu_cores': cpu_cores,
            'memory_limit': memory_limit,
            'gpu_sm_limit': gpu_sm_limit,
            'gpu_memory_limit': gpu_memory_limit,
            'execution_time': execution_time,
            'resource_cost': resource_cost,
            'objective_value': objective_value,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"执行时间: {execution_time:.2f}s, 资源成本: {resource_cost:.3f}, 目标值: {objective_value:.3f}")
        
        return objective_value
    
    def run_experiment(self, config):
        """运行单个实验配置"""
        try:
            # 使用第一个可用的图片文件
            image_files = self.experiment_executor.get_image_files()
            if not image_files:
                raise ValueError("没有找到图片文件")
            
            # 选择图片文件
            if self.image_filter:
                matching_images = [f for f in image_files if any(pattern in f for pattern in self.image_filter)]
                image_path = matching_images[0] if matching_images else image_files[0]
            else:
                image_path = image_files[0]
            
            # 运行实验
            results = self.experiment_executor.run_single_experiment(config, config['task'], image_path, repeat=1)
            
            if results and len(results) > 0 and results[0]['success']:
                return results[0]['docker_execution_time']
            else:
                logger.warning("实验失败，返回较大的惩罚值")
                return 1000.0  # 大的惩罚值
                
        except Exception as e:
            logger.error(f"运行实验失败: {e}")
            return 1000.0  # 大的惩罚值
    
    def optimize(self, n_calls=20, n_initial_points=None):
        """运行贝叶斯优化"""
        logger.info(f"开始贝叶斯优化，目标调用次数: {n_calls}")
        
        # 设置dimensions到objective_function
        self.objective_function.dimensions = self.dimensions
        
        # 如果没有指定初始点数量，使用种子数据的数量
        if n_initial_points is None:
            n_initial_points = min(len(self.initial_points), n_calls // 2)
        
        try:
            # 运行贝叶斯优化
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                x0=self.initial_points[:n_initial_points] if self.initial_points else None,
                y0=self.initial_values[:n_initial_points] if self.initial_values else None,
                acquisition_function='EI',  # Expected Improvement
                random_state=42,
                verbose=True
            )
            
            logger.info("贝叶斯优化完成")
            
            # 保存结果
            self.save_optimization_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"贝叶斯优化失败: {e}")
            raise
    
    def save_optimization_results(self, result):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存优化历史
        history_file = f"bayesian_optimization_history_{timestamp}.csv"
        history_df = pd.DataFrame(self.optimization_history)
        history_df.to_csv(history_file, index=False)
        logger.info(f"优化历史保存到: {history_file}")
        
        # 保存最优结果
        best_params = result.x
        best_value = result.fun
        
        best_result = {
            'best_cpu_cores': best_params[0],
            'best_memory_limit': best_params[1], 
            'best_gpu_sm_limit': best_params[2],
            'best_gpu_memory_limit': best_params[3],
            'best_objective_value': best_value,
            'n_calls': result.n_calls,
            'optimization_time': timestamp
        }
        
        # 保存为JSON
        result_file = f"bayesian_optimization_result_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(best_result, f, indent=2)
        logger.info(f"最优结果保存到: {result_file}")
        
        # 打印最优配置
        logger.info("=" * 50)
        logger.info("最优配置:")
        logger.info(f"  CPU cores: {best_params[0]}")
        logger.info(f"  Memory: {best_params[1]}")
        logger.info(f"  GPU SM limit: {best_params[2]}")
        logger.info(f"  GPU memory: {best_params[3]}")
        logger.info(f"  目标值: {best_value:.3f}")
        logger.info("=" * 50)
        
        # 生成推荐的配置文件
        best_config = self.convert_params_to_config(best_params)
        recommended_config = {
            'configurations': [best_config]
        }
        
        config_file = f"bayesian_optimized_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(recommended_config, f, indent=2)
        logger.info(f"推荐配置保存到: {config_file}")
    
    def analyze_convergence(self):
        """分析收敛情况"""
        if not self.optimization_history:
            return
        
        history_df = pd.DataFrame(self.optimization_history)
        
        # 计算累积最优值
        cumulative_best = []
        current_best = float('inf')
        
        for obj_val in history_df['objective_value']:
            if obj_val < current_best:
                current_best = obj_val
            cumulative_best.append(current_best)
        
        history_df['cumulative_best'] = cumulative_best
        
        # 打印收敛分析
        logger.info("收敛分析:")
        logger.info(f"  初始最优值: {cumulative_best[0]:.3f}")
        logger.info(f"  最终最优值: {cumulative_best[-1]:.3f}")
        logger.info(f"  改进幅度: {(cumulative_best[0] - cumulative_best[-1]) / cumulative_best[0] * 100:.1f}%")
        
        return history_df


def main():
    parser = argparse.ArgumentParser(description="基于贝叶斯优化的资源配置实验")
    parser.add_argument("--seed-data", required=True, help="种子数据CSV文件路径")
    parser.add_argument("--task", nargs="+", help="要优化的任务")
    parser.add_argument("--image", nargs="+", help="要使用的图片文件")
    parser.add_argument("--n-calls", type=int, default=20, help="贝叶斯优化调用次数")
    parser.add_argument("--n-initial", type=int, help="初始随机点数量")
    parser.add_argument("--docker-image", default="model-runner", help="Docker镜像名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建优化器
        optimizer = BayesianOptimizer(
            docker_image=args.docker_image,
            seed_data_file=args.seed_data,
            task_filter=args.task,
            image_filter=args.image
        )
        
        # 运行优化
        result = optimizer.optimize(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial
        )
        
        # 分析收敛
        optimizer.analyze_convergence()
        
        print("\n贝叶斯优化完成!")
        
    except KeyboardInterrupt:
        logger.info("优化被用户中断")
    except Exception as e:
        logger.error(f"优化失败: {e}")
        raise


if __name__ == "__main__":
    main()