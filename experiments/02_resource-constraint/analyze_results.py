#!/usr/bin/env python3
"""
实验结果分析脚本
用于分析和可视化资源约束实验的结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path
import json

class ExperimentAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载实验数据"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"成功加载数据: {len(self.df)} 条记录")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.csv_file}")
            raise
        except Exception as e:
            print(f"错误: 加载数据失败 - {e}")
            raise
    
    def basic_statistics(self):
        """基础统计信息"""
        print("\n=== 基础统计信息 ===")
        print(f"总实验数: {len(self.df)}")
        print(f"成功实验数: {len(self.df[self.df['success'] == True])}")
        print(f"失败实验数: {len(self.df[self.df['success'] == False])}")
        
        print(f"\n配置分布:")
        print(self.df['config_name'].value_counts())
        
        print(f"\n任务分布:")
        print(self.df['task'].value_counts())
        
        # 成功实验的延迟统计
        successful_df = self.df[self.df['success'] == True]
        if len(successful_df) > 0 and 'end_to_end_time' in successful_df.columns:
            print(f"\n延迟统计 (成功实验):")
            print(f"平均延迟: {successful_df['end_to_end_time'].mean():.3f} 秒")
            print(f"中位数延迟: {successful_df['end_to_end_time'].median():.3f} 秒")
            print(f"最小延迟: {successful_df['end_to_end_time'].min():.3f} 秒")
            print(f"最大延迟: {successful_df['end_to_end_time'].max():.3f} 秒")
            print(f"标准差: {successful_df['end_to_end_time'].std():.3f} 秒")
    
    def plot_latency_by_config(self, save_path=None):
        """按配置绘制延迟分布"""
        successful_df = self.df[self.df['success'] == True]
        if len(successful_df) == 0 or 'end_to_end_time' not in successful_df.columns:
            print("没有成功的实验数据或缺少延迟数据")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 按配置和任务分组的平均延迟
        avg_latency = successful_df.groupby(['config_name', 'task'])['end_to_end_time'].mean().reset_index()
        
        # 创建透视表
        pivot_table = avg_latency.pivot(index='config_name', columns='task', values='end_to_end_time')
        
        # 绘制热力图
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': '平均延迟 (秒)'})
        plt.title('不同配置和任务的平均延迟热力图')
        plt.xlabel('任务类型')
        plt.ylabel('资源配置')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_resource_impact(self, save_path=None):
        """分析资源限制对性能的影响"""
        successful_df = self.df[self.df['success'] == True]
        if len(successful_df) == 0 or 'end_to_end_time' not in successful_df.columns:
            print("没有成功的实验数据或缺少延迟数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('资源限制对性能的影响分析', fontsize=16)
        
        # CPU核心数 vs 延迟
        if 'cpu_cores' in successful_df.columns:
            cpu_data = successful_df.groupby('cpu_cores')['end_to_end_time'].mean()
            axes[0, 0].bar(range(len(cpu_data)), cpu_data.values)
            axes[0, 0].set_xlabel('CPU 核心数')
            axes[0, 0].set_ylabel('平均延迟 (秒)')
            axes[0, 0].set_title('CPU 核心数对延迟的影响')
            axes[0, 0].set_xticks(range(len(cpu_data)))
            axes[0, 0].set_xticklabels(cpu_data.index)
        
        # 内存限制 vs 延迟
        if 'memory_limit' in successful_df.columns:
            memory_data = successful_df.groupby('memory_limit')['end_to_end_time'].mean()
            axes[0, 1].bar(range(len(memory_data)), memory_data.values)
            axes[0, 1].set_xlabel('内存限制')
            axes[0, 1].set_ylabel('平均延迟 (秒)')
            axes[0, 1].set_title('内存限制对延迟的影响')
            axes[0, 1].set_xticks(range(len(memory_data)))
            axes[0, 1].set_xticklabels(memory_data.index, rotation=45)
        
        # GPU SM限制 vs 延迟
        if 'gpu_sm_limit' in successful_df.columns:
            gpu_sm_data = successful_df.groupby('gpu_sm_limit')['end_to_end_time'].mean()
            axes[1, 0].bar(range(len(gpu_sm_data)), gpu_sm_data.values)
            axes[1, 0].set_xlabel('GPU SM 限制')
            axes[1, 0].set_ylabel('平均延迟 (秒)')
            axes[1, 0].set_title('GPU SM 限制对延迟的影响')
            axes[1, 0].set_xticks(range(len(gpu_sm_data)))
            axes[1, 0].set_xticklabels(gpu_sm_data.index)
        
        # GPU内存限制 vs 延迟
        if 'gpu_memory_limit' in successful_df.columns:
            gpu_mem_data = successful_df.groupby('gpu_memory_limit')['end_to_end_time'].mean()
            axes[1, 1].bar(range(len(gpu_mem_data)), gpu_mem_data.values)
            axes[1, 1].set_xlabel('GPU 内存限制 (MB)')
            axes[1, 1].set_ylabel('平均延迟 (秒)')
            axes[1, 1].set_title('GPU 内存限制对延迟的影响')
            axes[1, 1].set_xticks(range(len(gpu_mem_data)))
            axes[1, 1].set_xticklabels(gpu_mem_data.index, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_memory_usage(self, save_path=None):
        """分析内存使用情况"""
        successful_df = self.df[self.df['success'] == True]
        if len(successful_df) == 0:
            print("没有成功的实验数据")
            return
        
        memory_cols = ['gpu_current_memory', 'gpu_peak_memory', 'cpu_memory']
        available_cols = [col for col in memory_cols if col in successful_df.columns]
        
        if not available_cols:
            print("没有内存使用数据")
            return
        
        fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 5))
        if len(available_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_cols):
            data = successful_df.groupby('task')[col].mean()
            axes[i].bar(data.index, data.values)
            axes[i].set_xlabel('任务类型')
            axes[i].set_ylabel('内存使用 (GB)')
            axes[i].set_title(f'{col.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def export_summary(self, output_file):
        """导出摘要报告"""
        successful_df = self.df[self.df['success'] == True]
        
        summary = {
            'experiment_overview': {
                'total_experiments': len(self.df),
                'successful_experiments': len(successful_df),
                'failure_rate': (len(self.df) - len(successful_df)) / len(self.df) * 100
            },
            'configuration_performance': {},
            'task_performance': {},
            'resource_efficiency': {}
        }
        
        if len(successful_df) > 0 and 'end_to_end_time' in successful_df.columns:
            # 按配置分析性能
            config_perf = successful_df.groupby('config_name')['end_to_end_time'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            summary['configuration_performance'] = config_perf
            
            # 按任务分析性能
            task_perf = successful_df.groupby('task')['end_to_end_time'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            summary['task_performance'] = task_perf
            
            # 资源效率分析
            if all(col in successful_df.columns for col in ['cpu_cores', 'memory_limit', 'gpu_sm_limit', 'gpu_memory_limit']):
                resource_eff = successful_df.groupby(['cpu_cores', 'memory_limit', 'gpu_sm_limit', 'gpu_memory_limit'])['end_to_end_time'].mean().to_dict()
                summary['resource_efficiency'] = resource_eff
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"摘要报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="实验结果分析脚本")
    parser.add_argument("csv_file", help="实验结果CSV文件路径")
    parser.add_argument("--output-dir", default="analysis_output", help="输出目录")
    parser.add_argument("--no-plots", action="store_true", help="不生成图表")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"错误: 找不到文件 {args.csv_file}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = ExperimentAnalyzer(args.csv_file)
    
    # 基础统计
    analyzer.basic_statistics()
    
    # 生成图表
    if not args.no_plots:
        print("\n正在生成分析图表...")
        
        try:
            analyzer.plot_latency_by_config(output_dir / "latency_heatmap.png")
            analyzer.plot_resource_impact(output_dir / "resource_impact.png")
            analyzer.plot_memory_usage(output_dir / "memory_usage.png")
        except Exception as e:
            print(f"生成图表时出错: {e}")
    
    # 导出摘要
    analyzer.export_summary(output_dir / "experiment_summary.json")
    
    print(f"\n分析完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()