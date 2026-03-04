# Quick Start Guide

## 数据增强快速使用指南

### 1. 查看原始数据统计

```bash
python scripts/analyze_tasks.py
```

**输出示例:**
- 总任务数: 866
- 总节点数: 931
- 大多数任务是孤立节点（in-degree=0, out-degree=0）

### 2. 生成增强数据

```bash
# 默认参数：生成1000个增强任务，每个合并3个子任务
python scripts/augment_tasks.py

# 自定义参数示例：生成500个增强任务，每个合并5个子任务
python scripts/augment_tasks.py --num-augmented 500 --merge-count 5

# 完整参数示例
python scripts/augment_tasks.py \
  --input data/tasks.json \
  --output data/tasks_augmented.json \
  --num-augmented 1000 \
  --merge-count 3 \
  --seed 42
```

**参数说明:**
- `--input`: 输入JSON文件路径（默认: `data/tasks.json`）
- `--output`: 输出JSON文件路径（默认: `data/tasks_augmented.json`）
- `--num-augmented`: 生成的增强任务数量（默认: 1000）
- `--merge-count`: 每个增强任务合并的子任务数量（默认: 3）
- `--seed`: 随机种子，用于结果可复现（默认: 42）

### 3. 查看增强数据统计

```bash
python scripts/analyze_augmented.py
```

**输出包括:**
- 任务类型分布
- 合并任务的节点和链接统计
- 节点度数分布
- 示例任务展示

### 4. 查看示例任务

```bash
python scripts/show_samples.py
```

**展示内容:**
- 原始任务（带START节点）
- 链式任务（带START节点）
- 简单合并任务
- 复杂合并任务
- 复杂度分布统计

## 数据增强原理

### 步骤1: 为所有任务添加START节点

原始任务:
```
[Task A] -> [Task B]
```

添加START后:
```
[START] -> [Task A] -> [Task B]
```

### 步骤2: 合并多个任务图

选择3个任务并合并:

**子任务1:**
```
[Task A]
```

**子任务2:**
```
[Task B] -> [Task C]
```

**子任务3:**
```
[Task D]
```

**合并后:**
```
              ┌─> [T1_Task A]
              │
[START] ─────┼─> [T2_Task B] -> [T2_Task C]
              │
              └─> [T3_Task D]
```

### 关键特性

1. **唯一命名**: 每个子任务的节点都添加前缀（T1_, T2_, T3_等）避免命名冲突
2. **保留原始结构**: 每个子任务内部的DAG结构保持不变
3. **START节点**: 统一的入口点，连接到所有子图的根节点
4. **元数据保留**: 记录原始任务ID，便于追溯
5. **格式化指令**: 合并任务的 instruction 使用数字列表（1. 2. 3.）清晰展示多个子任务

### 步骤3: 格式化合并任务的指令

**原始3个任务的指令:**
- 任务1: "I need to translate this text..."
- 任务2: "Can you detect objects in example.jpg?"
- 任务3: "Please classify this image..."

**合并后的指令格式:**
```
1. I need to translate this text...

2. Can you detect objects in example.jpg?

3. Please classify this image...
```

这种格式使多步骤工作流更加清晰易读。

## 生成的数据格式

增强后的数据文件包含:

```json
{
  "data": [
    // 866个原始任务（添加了START节点）
    {
      "id": "27766469",
      "type": "single_with_start",
      "n_tools": 2,
      "tool_nodes": [...],
      "tool_links": [...]
    },
    // 1000个合并任务
    {
      "id": "AUG_00000001",
      "type": "merged",
      "n_tools": 4,
      "original_task_ids": ["task1_id", "task2_id", "task3_id"],
      "instruction": "1. First task instruction...\n\n2. Second task instruction...\n\n3. Third task instruction...",
      "tool_nodes": [...],
      "tool_links": [...]
    }
  ],
  "metadata": {
    "original_count": 866,
    "augmented_count": 1000,
    "total_count": 1866,
    "merge_count": 3
  }
}
```

## 数据统计对比

### 原始数据 (data/tasks.json)
- 任务数: 866
- 节点度数: 大部分是(0,0)
- 平均节点数: ~1.07

### 增强数据 (data/tasks_augmented.json)
- 任务数: 1866 (866原始 + 1000合并)
- 合并任务平均节点数: ~4.23
- 合并任务平均链接数: ~3.23
- 节点度数: 更多样化的分布

## 常见用例

### 用例1: 生成小规模测试数据
```bash
python scripts/augment_tasks.py --num-augmented 10 --output data/test.json
```

### 用例2: 生成高复杂度数据
```bash
python scripts/augment_tasks.py --merge-count 5 --num-augmented 500
```

### 用例3: 批量生成不同配置
```bash
# 合并2个任务
python scripts/augment_tasks.py --merge-count 2 --output data/aug_m2.json

# 合并3个任务
python scripts/augment_tasks.py --merge-count 3 --output data/aug_m3.json

# 合并4个任务
python scripts/augment_tasks.py --merge-count 4 --output data/aug_m4.json
```

## 生成执行计划 (Ground Truth)

使用 `scripts/generate_execution_plans.py` 为每个任务生成带资源调度的执行计划，作为训练数据。

### 基本用法

```bash
python scripts/generate_execution_plans.py \
  --tasks data/tasks_augmented.json \
  --system data/system.json \
  --profiling data/profiling.csv \
  --output data/execution_plans.json \
  --scenarios 3 \
  --seed 42
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | — | 增强后的任务 JSON 文件 |
| `--system` | — | 系统资源上限配置 (`system.json`) |
| `--profiling` | — | 工具性能矩阵 (`profiling.csv`) |
| `--output` | — | 输出执行计划 JSON 路径 |
| `--scenarios` | 3 | 每个任务生成的资源场景数 |
| `--batch-dir` | 无 | 批量输出目录（可选） |
| `--batch-size` | 100 | 每批文件包含的任务数 |
| `--workers` | 1 | 并行进程数（多核时使用 8） |
| `--max-tasks` | 全量 | 限制处理任务数（调试用） |
| `--task-filter` | all | 过滤任务类型：`all/single/multi/merged` |
| `--add-parallel` | 关闭 | 为 merged/dag 任务额外生成强制并行计划 |

### `--add-parallel` 说明

merged 任务在常规资源场景下始终串行（串行高配 < 并行低配的延迟），训练数据缺乏并行样本，导致模型无法学会"资源充足时并行调度"。

启用 `--add-parallel` 后，每个 merged/dag 任务会额外生成 **1 个强制并行计划**：
- `SYSTEM_STATE` = 系统满配资源
- 子任务按数量 **均分** 系统资源（1/N 份）
- 所有独立子任务 **同时启动**，无 `<WAIT>`

**对比样本示例：**

```
# 资源受限场景（串行）
SYSTEM_STATE: {cpu_core: 10, gpu_sm: 68}
<REF_0> = <EXEC> <TOOL_A>(...)
<REF_1> = <WAIT> <REF_0> <EXEC> <TOOL_B>(...)
<FINISH> <REF_0> <REF_1>

# 资源充足场景（并行，--add-parallel 生成）
SYSTEM_STATE: {cpu_core: 16, gpu_sm: 100}
<REF_0> = <EXEC> <TOOL_A>(...)
<REF_1> = <EXEC> <TOOL_B>(...)
<FINISH> <REF_0> <REF_1>
```

### 完整生产用法

```bash
# 生成全量执行计划（含并行对比数据）
python scripts/generate_execution_plans.py \
  --tasks data/tasks_augmented.json \
  --system data/system.json \
  --profiling data/profiling.csv \
  --output data/execution_plans_stage1.json \
  --batch-dir data/gt_stage1 \
  --batch-size 100 \
  --scenarios 3 \
  --workers 8 \
  --seed 42 \
  --add-parallel

# 快速验证（单任务）
python scripts/generate_execution_plans.py \
  --tasks data/tasks_augmented.json \
  --system data/system.json \
  --profiling data/profiling.csv \
  --output /tmp/plans.json \
  --max-tasks 1 \
  --workers 1 \
  --scenarios 1
```

## 编程接口使用

```python
from src.models import ToolDAG
import json

# 加载增强数据
with open('data/tasks_augmented.json', 'r') as f:
    data = json.load(f)

# 筛选合并任务
merged_tasks = [t for t in data['data'] if t['type'] == 'merged']

# 分析单个任务
task = ToolDAG.from_dict(merged_tasks[0])
print(f"Task ID: {task.id}")
print(f"Nodes: {len(task.tool_nodes)}")
print(f"Links: {len(task.tool_links)}")

# 获取节点度数
degrees = task.get_node_degrees()
for node_name, (in_deg, out_deg) in degrees.items():
    print(f"{node_name}: in={in_deg}, out={out_deg}")
```
