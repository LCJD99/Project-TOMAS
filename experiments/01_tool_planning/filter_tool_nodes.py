#!/usr/bin/env python3

import json
import sys
import click
from typing import List, Dict, Any


def load_json_file(file_path: str) -> Any:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON文件 {file_path} 时出错: {e}")
        sys.exit(1)


def save_json_file(data: Any, file_path: str) -> None:
    """保存数据到JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"筛选结果已保存到: {file_path}")
    except Exception as e:
        print(f"错误: 保存文件 {file_path} 时出错: {e}")
        sys.exit(1)


def extract_tool_tasks(tool_nodes: List[Dict]) -> List[str]:
    """从tool_nodes中提取任务名称列表"""
    tasks = []
    for node in tool_nodes:
        if isinstance(node, dict) and 'task' in node:
            tasks.append(node['task'])
        else:
            print(f"警告: 跳过异常的tool_node: {node}")
    return tasks


def group_data_by_tool_combinations(data: List[Dict], tool_combinations: List[List[str]]) -> List[Dict]:
    """根据工具组合分组数据，返回符合load_tasks_iterator期望的格式"""
    grouped_data = []
    
    # 为每个工具组合创建一个组
    for batch_index, combination in enumerate(tool_combinations):
        combination_set = set(combination)
        batch_data = []
        
        for item in data:
            if 'tool_nodes' not in item:
                continue
            
            # 排除type为"single"的数据
            if item.get('type') == 'single':
                continue
            
            # 检查tool_nodes是否为列表
            if not isinstance(item['tool_nodes'], list):
                continue
                
            # 提取工具任务名称
            tool_tasks = extract_tool_tasks(item['tool_nodes'])
            
            # 如果没有有效的任务，跳过
            if not tool_tasks:
                continue
            
            # 检查工具任务是否为该组合的子集
            tool_tasks_set = set(tool_tasks)
            if tool_tasks_set.issubset(combination_set):
                batch_data.append(item)
        
        # 如果该组合有匹配的数据且任务数>=5，添加到结果中
        if batch_data and len(batch_data) >= 5:
            grouped_data.append({
                "batch_index": batch_index,
                "combination": combination,
                "data": batch_data
            })
            print(f"批次 {batch_index}: 组合 {combination}, 匹配 {len(batch_data)} 个任务")
        elif batch_data and len(batch_data) < 5:
            print(f"批次 {batch_index}: 组合 {combination}, 匹配 {len(batch_data)} 个任务 (少于5个，已跳过)")
    
    return grouped_data


@click.command()
@click.option('--data_file', default='data_parsed.json', type=str, help='Path to the data_parsed.json file.')
@click.option('--batch_node_file', default='batch_node.json', type=str, help='Path to the batch_node.json file.')
@click.option('--output_file', default='filtered_data.json', type=str, help='Path to save the filtered_data.json file.')
def main(data_file, batch_node_file, output_file):
    """主函数"""
    # 文件路径
    batch_node_file = batch_node_file
    data_file = data_file
    output_file = output_file

    # 加载数据
    tool_combinations = load_json_file(batch_node_file)
    data = load_json_file(data_file)
    
    print(f"工具组合数量: {len(tool_combinations)}")
    print("工具组合:")
    for i, combination in enumerate(tool_combinations, 1):
        print(f"  组合 {i}: {combination}")
    
    print(f"\n原始数据项数量: {len(data)}")
    
    # 统计type为single的数据项
    single_count = sum(1 for item in data if item.get('type') == 'single')
    print(f"其中type为'single'的数据项: {single_count} (已排除)")
    
    # 分组数据
    grouped_data = group_data_by_tool_combinations(data, tool_combinations)
    
    # 计算跳过的batch数量
    skipped_batches = len(tool_combinations) - len(grouped_data)
    
    # 计算总的匹配项数量
    total_filtered = sum(len(group['data']) for group in grouped_data)
    print(f"\n筛选后数据项数量: {total_filtered}")
    print(f"分组数量: {len(grouped_data)}")
    print(f"跳过的批次数量: {skipped_batches} (任务数<5)")
    
    # 保存结果
    if grouped_data:
        save_json_file(grouped_data, output_file)
        
        # 显示统计信息
        print("\n=== 筛选统计 ===")
        print(f"原始数据项: {len(data)}")
        print(f"匹配数据项: {total_filtered}")
        print(f"匹配率: {total_filtered/len(data)*100:.2f}%")
        print(f"生成批次数: {len(grouped_data)}")
        print(f"跳过批次数: {skipped_batches} (任务数<5)")
        print(f"总工具组合数: {len(tool_combinations)}")
        
        print("\n=== 各批次详情 ===")
        for group in grouped_data:
            print(f"  批次 {group['batch_index']}: {group['combination']} - {len(group['data'])} 个任务")
    else:
        print("没有找到匹配的数据项")


if __name__ == "__main__":
    main()
