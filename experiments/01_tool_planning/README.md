# 工具规划与评估实验

本项目是一个用于工具规划(Tool Planning)的实验框架，主要目的是研究大模型在多任务场景下的工具选择、组合和规划能力。项目通过对比大模型生成的工具规划方案与真实标准(Ground Truth)，使用F1指标评估模型的性能。

## 项目概述

该实验框架模拟了一个多工具环境，其中每个工具具有特定的功能（如文本生成、图像分割、机器翻译等）。给定一组任务，系统需要：

1. 选择合适的工具
2. 规划工具的执行顺序
3. 优化工具复用以提高效率
4. 生成符合依赖关系的执行计划

## 文件说明

### 核心脚本

- **`sample_tools.py`** - 工具组合采样器
  - 从所有可用工具中随机抽取多个工具组合
  - 生成不同规模的工具集合用于实验
  - 输出：`batch_node.json`

- **`filter_tool_nodes.py`** - 任务筛选器
  - 根据指定的工具组合筛选符合条件的任务请求
  - 过滤掉不匹配的任务，确保实验数据质量
  - 输入：`data_parsed.json`, `batch_node.json`
  - 输出：`filtered_data.json`

- **`inference.py`** - 大模型推理引擎
  - 调用大语言模型生成结构化的工具规划方案
  - 支持零样本(Zero-shot)和少样本(Few-shot)推理
  - 输出格式化的JSON工具执行计划
  - 输出：`zero_shot_response.json`, `few_shot_response.json`

- **`evaluation.py`** - 性能评估分析器
  - 使用F1指标评估大模型规划结果与真实标准的差异
  - 分析工具复用模式和执行效率
  - 生成详细的性能分析报告
  - 支持可视化任务依赖图

### 数据文件

- **`tool_desc.json`** - 工具描述文件，包含所有可用工具的定义和功能描述
- **`batch_node.json`** - 采样的工具组合列表
- **`data_parsed.json`** - 原始任务数据集
- **`filtered_data.json`** - 筛选后的匹配任务数据
- **`*_response.json`** - 大模型生成的工具规划结果
- **`*_prompts.txt`** - 发送给大模型的提示词记录

## 使用方法

### 1. 生成工具组合

```bash
python sample_tools.py --input_data ./tool_desc.json --output_file batch_node.json
```

### 2. 筛选匹配任务

```bash
python filter_tool_nodes.py --data_file data_parsed.json --batch_node_file batch_node.json --output_file filtered_data.json
```

### 3. 运行大模型推理

零样本推理：
```bash
python inference.py --workload 2 --seed 42
```

少样本推理：
```bash
python inference.py --workload 2 --few-shot few_shot_example.txt --seed 42
```

### 4. 评估结果

基础评估：
```bash
python evaluation.py --data_path zero_shot_response.json --ground_truth filtered_data.json
```

包含工具复用分析：
```bash
python evaluation.py --data_path zero_shot_response.json --ground_truth filtered_data.json --analyze_reuse
```

## 实验流程

1. **工具采样**: 从完整工具集中随机选择工具子集
2. **任务筛选**: 根据工具组合筛选相关任务
3. **模型推理**: 大模型生成工具规划方案
4. **性能评估**: 对比生成方案与标准答案，计算F1分数
5. **结果分析**: 分析工具复用模式和执行效率

## 评估指标

- **工具选择F1**: 评估模型选择正确工具的能力
- **工具复用F1**: 评估模型识别和利用工具复用机会的能力
- **执行计划准确性**: 评估生成的任务依赖关系的正确性

## 依赖要求

- Python 3.7+
- OpenAI API (用于大模型调用)
- NetworkX (用于图分析)
- Matplotlib (用于可视化)
- scikit-learn (用于评估指标)
- Click (用于命令行接口)

## 注意事项

- 确保在运行推理前已正确配置OpenAI API
- 建议使用固定的随机种子以确保实验可重现性
- 大规模实验可能需要较长的推理时间