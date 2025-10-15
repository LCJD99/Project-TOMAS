# 资源约束实验

## 实验概述

本实验用于测试不同资源约束（CPU、内存、GPU、GPU内存）下机器学习模型的性能表现。实验支持多种AI任务，包括图像超分辨率、图像描述和目标检测，通过动态生成Kubernetes YAML配置文件来测试不同资源配置组合的影响。

### 核心特性
- **多维资源约束**：支持CPU、内存、GPU、GPU内存的精确控制
- **vGPU支持**：利用HAMi调度器实现GPU资源的细粒度分配
- **多任务测试**：涵盖三种典型AI推理任务
- **自动化配置**：通过JSON配置文件和Jinja2模板批量生成测试场景

## 实验组件

### 资源配置生成器

#### generate_yamls.py
- **功能**：基于Jinja2模板批量生成Kubernetes Pod YAML配置
- **配置文件**：`resource_configurations.json` - 定义多种资源配置组合
- **模板文件**：`template.yaml.j2` - Kubernetes Pod配置模板
- **输出**：生成的YAML文件存放在独立目录，支持单独或批量任务配置

#### 资源配置类型
- **Small Config**：轻量级配置（0.5 CPU, 1Gi内存, 0.5 GPU, 2GB GPU内存）
- **Medium Config**：标准配置（1 CPU, 2Gi内存, 1.0 GPU, 4GB GPU内存）
- **Large Config**：重型配置（2 CPU, 4Gi内存, 1.0 GPU, 6GB GPU内存）
- **GPU Intensive**：GPU密集型配置（高GPU内存分配）
- **CPU Intensive**：CPU密集型配置（多核心分配）

### AI推理任务

#### inference.py
- **图像超分辨率**：使用Swin2SR模型进行图像增强
- **图像描述生成**：基于ViT-GPT2的图像标注
- **目标检测**：采用DETR模型进行物体识别

每个任务支持独立的资源配置和性能监控。

### Kubernetes部署

#### demo.yaml / template.yaml.j2
- 基于HAMi调度器的Pod配置模板
- 支持GPU资源的精确分割（nvidia.com/gpumem参数）
- 自动挂载Hugging Face缓存和数据卷
- 环境变量配置支持模型缓存和GPU可见性

## 子实验

### 1. CPU资源约束实验

#### 实验目标
测试CPU核心数和内存限制对模型推理性能的影响。
#### Docker容器组件

**Dockerfile**
- 基于Python运行环境构建容器镜像
- 安装所需依赖包（requirements.txt）
- 配置Python推理脚本（inference.py）作为入口点
- 设置环境变量以支持GPU和缓存

**run.sh脚本**
实验脚本包含两个主要测试函数：

1. **CPU核心数测试（cpu_core）**

- 测试不同CPU核心数：1, 2, 4, 8, 16, 32
- 在三个AI任务上进行测试：超分辨率、图像描述、目标检测
- 使用`--cpus`参数限制容器可用的CPU核心数

2. **CPU内存测试（cpu_memory）**

- 测试不同内存限制：512m, 800m, 1g, 1.5g, 2g
- 专注于目标检测任务
- 使用`-m`参数限制容器内存，`--memory-swap`设置交换内存

#### 运行方式
执行`./run.sh`脚本开始CPU资源约束测试，脚本会自动循环测试不同的资源配置组合。

### 2. GPU资源约束实验

#### 实验目标
利用HAMi调度器测试GPU资源分片对模型推理性能的影响。

#### 核心特性
- **vGPU资源管理**：通过`nvidia.com/gpumem`参数精确控制GPU内存分配
- **多配置测试**：自动生成多种GPU资源配置的Pod定义
- **性能对比**：不同GPU内存配置下的推理时间和资源利用率分析

## 快速开始

### 1. 生成配置文件

利用 `jinja2` 模板生成多种资源配置的YAML配置

```bash
# 生成通用资源配置的YAML文件
python generate_yamls.py

# 为每个任务单独生成YAML文件
python generate_yamls.py --all-tasks

# 指定输出目录
python generate_yamls.py --output my_configs
```

### 2. 部署测试Pod
```bash
# 部署特定配置的Pod
kubectl apply -f generated_yamls/medium-config_imagecaptioning.yaml

# 查看Pod状态
kubectl get pods -l experiment=resource-constraint-experiment
```

### 3. CPU约束测试
```bash
# 运行Docker CPU约束测试
./run.sh
```

## 配置文件说明

- `resource_configurations.json`：资源配置定义文件
- `template.yaml.j2`：Kubernetes Pod模板
- `demo.yaml`：示例配置文件
- `requirements.txt`：Python依赖包列表