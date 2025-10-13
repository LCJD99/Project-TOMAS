#!/usr/bin/env python3
"""
K8s Resource Constraint Experiment Results Analyzer
This script processes experiment results and generates comprehensive CSV reports
"""

import pandas as pd
import json
import os
import glob
from datetime import datetime
import argparse
import numpy as np

class ExperimentAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.csv_file = os.path.join(results_dir, "experiment_results.csv")
        self.analysis_file = os.path.join(results_dir, "experiment_analysis.csv")
        
    def load_results(self):
        """Load experiment results from CSV file"""
        if not os.path.exists(self.csv_file):
            print(f"错误: 结果文件 {self.csv_file} 不存在")
            return None
            
        try:
            df = pd.read_csv(self.csv_file)
            print(f"成功加载 {len(df)} 条实验结果")
            return df
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            return None
    
    def process_json_results(self):
        """Process individual JSON result files if available"""
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        results = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"处理JSON文件 {json_file} 失败: {e}")
                
        return results
    
    def analyze_performance(self, df):
        """Analyze performance metrics across different resource configurations"""
        if df is None or df.empty:
            return None
            
        # Group by resource configuration
        analysis = []
        
        # Basic statistics
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            
            for _, row in task_df.iterrows():
                analysis_row = {
                    'pod_name': row['pod_name'],
                    'task': row['task'],
                    'cpu_cores': row['cpu_cores'],
                    'memory_gb': row['memory_gb'],
                    'gpu_ratio': row['gpu_ratio'],
                    'gpu_memory_mb': row['gpu_memory_mb'],
                    'total_time': row['total_time'],
                    'task_execution_time': row['task_execution_time'],
                    'gpu_memory_used': row['gpu_memory_used'],
                    'cpu_memory_percent': row['cpu_memory_percent'],
                    'status': row['status'],
                    
                    # Performance metrics
                    'cpu_efficiency': self.calculate_cpu_efficiency(row),
                    'memory_efficiency': self.calculate_memory_efficiency(row),
                    'gpu_efficiency': self.calculate_gpu_efficiency(row),
                    'overall_efficiency': self.calculate_overall_efficiency(row),
                    
                    # Resource utilization ratios
                    'time_per_cpu_core': row['total_time'] / row['cpu_cores'] if row['cpu_cores'] > 0 else 0,
                    'time_per_gb_memory': row['total_time'] / row['memory_gb'] if row['memory_gb'] > 0 else 0,
                    'time_per_gpu_ratio': row['total_time'] / row['gpu_ratio'] if row['gpu_ratio'] > 0 else 0,
                    
                    'timestamp': row['timestamp']
                }
                
                analysis.append(analysis_row)
        
        return pd.DataFrame(analysis)
    
    def calculate_cpu_efficiency(self, row):
        """Calculate CPU efficiency based on execution time and CPU cores"""
        if row['cpu_cores'] > 0 and row['total_time'] > 0:
            # Lower time per core indicates higher efficiency 
            return 1.0 / (row['total_time'] / row['cpu_cores'])
        return 0
    
    def calculate_memory_efficiency(self, row):
        """Calculate memory efficiency"""
        if row['memory_gb'] > 0 and row['cpu_memory_percent'] > 0:
            # Higher memory usage percentage might indicate better utilization
            return row['cpu_memory_percent'] / 100.0
        return 0
    
    def calculate_gpu_efficiency(self, row):
        """Calculate GPU efficiency"""
        if row['gpu_ratio'] > 0 and row['gpu_memory_used'] > 0:
            # Efficiency based on GPU memory utilization
            gpu_memory_total_gb = (row['gpu_memory_mb'] / 1000.0) * row['gpu_ratio']
            if gpu_memory_total_gb > 0:
                return row['gpu_memory_used'] / gpu_memory_total_gb
        return 0
    
    def calculate_overall_efficiency(self, row):
        """Calculate overall system efficiency"""
        cpu_eff = self.calculate_cpu_efficiency(row)
        mem_eff = self.calculate_memory_efficiency(row)
        gpu_eff = self.calculate_gpu_efficiency(row)
        
        # Weighted average (you can adjust weights based on your priorities)
        return (cpu_eff * 0.4 + mem_eff * 0.2 + gpu_eff * 0.4)
    
    def generate_summary_stats(self, df):
        """Generate summary statistics"""
        if df is None or df.empty:
            return None
            
        summary = {}
        
        # Overall statistics
        summary['total_experiments'] = len(df)
        summary['successful_experiments'] = len(df[df['status'] == 'success'])
        summary['failed_experiments'] = len(df[df['status'] == 'failed'])
        summary['success_rate'] = summary['successful_experiments'] / summary['total_experiments'] * 100
        
        # Task-specific statistics
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            summary[f'{task}_avg_time'] = task_df['total_time'].mean()
            summary[f'{task}_min_time'] = task_df['total_time'].min()
            summary[f'{task}_max_time'] = task_df['total_time'].max()
            summary[f'{task}_std_time'] = task_df['total_time'].std()
        
        # Resource utilization statistics
        summary['avg_cpu_usage'] = df['cpu_memory_percent'].mean()
        summary['avg_gpu_usage'] = df['gpu_memory_used'].mean()
        
        return summary
    
    def find_optimal_configurations(self, analysis_df):
        """Find optimal resource configurations for each task"""
        if analysis_df is None or analysis_df.empty:
            return None
            
        optimal_configs = {}
        
        for task in analysis_df['task'].unique():
            task_df = analysis_df[analysis_df['task'] == task]
            
            # Find configuration with best overall efficiency
            best_efficiency_idx = task_df['overall_efficiency'].idxmax()
            best_efficiency_config = task_df.loc[best_efficiency_idx]
            
            # Find configuration with minimum execution time
            best_time_idx = task_df['total_time'].idxmin()
            best_time_config = task_df.loc[best_time_idx]
            
            optimal_configs[task] = {
                'best_efficiency': {
                    'pod_name': best_efficiency_config['pod_name'],
                    'cpu_cores': best_efficiency_config['cpu_cores'],
                    'memory_gb': best_efficiency_config['memory_gb'],
                    'gpu_ratio': best_efficiency_config['gpu_ratio'],
                    'gpu_memory_mb': best_efficiency_config['gpu_memory_mb'],
                    'total_time': best_efficiency_config['total_time'],
                    'overall_efficiency': best_efficiency_config['overall_efficiency']
                },
                'best_time': {
                    'pod_name': best_time_config['pod_name'],
                    'cpu_cores': best_time_config['cpu_cores'],
                    'memory_gb': best_time_config['memory_gb'],
                    'gpu_ratio': best_time_config['gpu_ratio'],
                    'gpu_memory_mb': best_time_config['gpu_memory_mb'],
                    'total_time': best_time_config['total_time'],
                    'overall_efficiency': best_time_config['overall_efficiency']
                }
            }
            
        return optimal_configs
    
    def save_analysis(self, analysis_df, summary_stats, optimal_configs):
        """Save analysis results to files"""
        
        # Save detailed analysis to CSV
        if analysis_df is not None:
            analysis_df.to_csv(self.analysis_file, index=False)
            print(f"详细分析结果已保存到: {self.analysis_file}")
        
        # Save summary statistics
        summary_file = os.path.join(self.results_dir, "experiment_summary.json")
        if summary_stats:
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            print(f"汇总统计已保存到: {summary_file}")
        
        # Save optimal configurations
        optimal_file = os.path.join(self.results_dir, "optimal_configurations.json")
        if optimal_configs:
            with open(optimal_file, 'w') as f:
                json.dump(optimal_configs, f, indent=2, default=str)
            print(f"最优配置已保存到: {optimal_file}")
    
    def print_summary(self, summary_stats, optimal_configs):
        """Print summary to console"""
        print("\n=== 实验结果汇总 ===")
        
        if summary_stats:
            print(f"总实验数: {summary_stats['total_experiments']}")
            print(f"成功实验数: {summary_stats['successful_experiments']}")
            print(f"失败实验数: {summary_stats['failed_experiments']}")
            print(f"成功率: {summary_stats['success_rate']:.1f}%")
            print(f"平均CPU使用率: {summary_stats.get('avg_cpu_usage', 0):.1f}%")
            print(f"平均GPU内存使用: {summary_stats.get('avg_gpu_usage', 0):.2f}GB")
        
        print("\n=== 各任务平均执行时间 ===")
        tasks = ['SuperResolution', 'ImageCaptioning', 'ObjectDetection']
        for task in tasks:
            avg_time = summary_stats.get(f'{task}_avg_time', 0)
            min_time = summary_stats.get(f'{task}_min_time', 0)
            max_time = summary_stats.get(f'{task}_max_time', 0)
            print(f"{task}: 平均 {avg_time:.2f}s, 最小 {min_time:.2f}s, 最大 {max_time:.2f}s")
        
        if optimal_configs:
            print("\n=== 最优配置推荐 ===")
            for task, configs in optimal_configs.items():
                print(f"\n{task}:")
                print(f"  最高效率配置: {configs['best_efficiency']['pod_name']}")
                print(f"    - CPU: {configs['best_efficiency']['cpu_cores']} cores")
                print(f"    - Memory: {configs['best_efficiency']['memory_gb']} GB")
                print(f"    - GPU: {configs['best_efficiency']['gpu_ratio']} ratio")
                print(f"    - 执行时间: {configs['best_efficiency']['total_time']:.2f}s")
                print(f"    - 效率分数: {configs['best_efficiency']['overall_efficiency']:.3f}")
                
                print(f"  最快执行配置: {configs['best_time']['pod_name']}")
                print(f"    - CPU: {configs['best_time']['cpu_cores']} cores")
                print(f"    - Memory: {configs['best_time']['memory_gb']} GB")
                print(f"    - GPU: {configs['best_time']['gpu_ratio']} ratio")
                print(f"    - 执行时间: {configs['best_time']['total_time']:.2f}s")
                print(f"    - 效率分数: {configs['best_time']['overall_efficiency']:.3f}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("开始分析实验结果...")
        
        # Load results
        df = self.load_results()
        if df is None:
            return
        
        # Run analysis
        analysis_df = self.analyze_performance(df)
        summary_stats = self.generate_summary_stats(df)
        optimal_configs = self.find_optimal_configurations(analysis_df)
        
        # Save results
        self.save_analysis(analysis_df, summary_stats, optimal_configs)
        
        # Print summary
        self.print_summary(summary_stats, optimal_configs)
        
        print(f"\n分析完成! 结果文件保存在: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze K8s resource constraint experiment results')
    parser.add_argument('--results-dir', 
                       default='/Users/lcjd/code-workspace/project/Project-TOMAS/experiments/02_resource-constraint/k8s_results',
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 {args.results_dir} 不存在")
        return
    
    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()