# Kubernetes Resource Constraint Experiments

这个目录包含基于 Kubernetes 的资源约束实验系统，用于评估在不同 CPU Core、CPU Memory、GPU 和 GPU Memory 约束下的模型推理性能。

## 系统架构

本实验系统采用模板化配置生成和自动化执行流程：

1. **配置生成**: 使用 Jinja2 模板和 JSON 配置生成多个 K8s Pod YAML 文件
2. **实验执行**: 自动创建和管理 K8s Pod，收集执行结果
3. **结果分析**: 将结果导出为 CSV 格式，支持进一步分析

## 文件结构

```
02_resource-constraint/
├── README_K8S.md                    # 本文档
├── resource_configs.json           # 资源约束配置文件
├── k8s-pod-template.yaml.j2        # Jinja2 Pod 模板
├── generate_k8s_configs.py         # 配置生成脚本
├── k8s_inference.py                # K8s 推理执行脚本
├── run_k8s_experiments.py          # 主实验运行脚本
├── inference.py                    # 原始推理脚本
├── lena.png                        # 测试图像
├── gpu-pod.yaml                    # 原始 GPU Pod 配置示例
└── k8s-configs/                    # 生成的配置文件目录（运行后生成）
    ├── experiment_manifest.json        # 实验清单
    └── inference-*.yaml                # 各个实验的 Pod 配置
```

## 资源约束配置

实验包含多种资源约束组合，针对您的 RTX 3090 GPU (24GB VRAM)：

| 配置名称 | CPU 核心 | CPU 内存 | GPU 分配 | GPU 内存 | 说明 |
|----------|----------|----------|----------|----------|------|
| low-resource | 1 | 1Gi | 0.25 | 6000Mi | 低资源配置 |
| medium-resource-1 | 2 | 2Gi | 0.5 | 12000Mi | 中等资源配置-1 |
| medium-resource-2 | 4 | 4Gi | 0.5 | 12000Mi | 中等资源配置-2 |
| high-resource-1 | 8 | 8Gi | 1.0 | 24000Mi | 高资源配置-1 |
| high-resource-2 | 16 | 16Gi | 1.0 | 24000Mi | 高资源配置-2 |
| cpu-constrained | 1 | 8Gi | 1.0 | 24000Mi | CPU 核心受限 |
| memory-constrained | 8 | 1Gi | 1.0 | 24000Mi | CPU 内存受限 |
| gpu-memory-constrained | 8 | 8Gi | 1.0 | 6000Mi | GPU 内存受限 |
| gpu-fraction-constrained | 8 | 8Gi | 0.25 | 24000Mi | GPU 分配受限 |

## 测试任务

每个资源配置将测试 3 种 AI 任务：
- **SuperResolution**: 图像超分辨率 (Swin2SR-classical-sr-x2-64)
- **ImageCaptioning**: 图像描述生成 (ViT-GPT2)
- **ObjectDetection**: 目标检测 (DETR-ResNet-101)

总计：9 种资源配置 × 3 种任务 = **27 个实验**

## 使用方法

### 前置条件

1. **Kubernetes集群**: 确保您有一个可用的K8s集群，并且`kubectl`已正确配置
2. **HAMI调度器**: 确保集群中安装了HAMI调度器以支持vGPU
3. **镜像**: 确保`docker.cnb.cool/lcjd1024/os`镜像可用且包含所需依赖
4. **HuggingFace缓存**: 确保模型文件已缓存到`~/.cache/huggingface`

### 运行实验

1. **检查配置**:
   ```bash
   # 验证kubectl连接
   kubectl cluster-info
   
   # 检查HAMI调度器
   kubectl get pods -n kube-system | grep hami
   ```

2. **执行实验**:
   ```bash
   # 进入实验目录
   cd /Users/lcjd/code-workspace/project/Project-TOMAS/experiments/02_resource-constraint
   
   # 让脚本可执行
   chmod +x run_k8s_experiments.sh
   
   # 运行所有实验 (约15个实验，每个5-10分钟)
   ./run_k8s_experiments.sh
   ```

3. **分析结果**:
   ```bash
   # 安装分析所需的Python包
   pip install pandas numpy
   
   # 运行结果分析
   python analyze_results.py
   ```

### 监控实验进展

实验运行时，您可以使用以下命令监控：

```bash
# 查看当前运行的Pod
kubectl get pods | grep gpu-experiment

# 查看特定Pod的日志
kubectl logs <pod-name> -f

# 查看Pod的资源使用情况
kubectl top pod <pod-name>
```

## 输出结果

### CSV结果文件 (`experiment_results.csv`)

包含每个实验的详细指标：
- `pod_name`: Pod配置名称
- `task`: 执行的任务类型
- `cpu_cores`: CPU核心数
- `memory_gb`: CPU内存限制(GB)
- `gpu_ratio`: GPU分配比例
- `gpu_memory_mb`: GPU内存限制(MB)
- `total_time`: 总执行时间(秒)
- `task_execution_time`: 任务执行时间(秒)
- `gpu_memory_used`: 实际GPU内存使用(GB)
- `cpu_memory_percent`: CPU内存使用百分比
- `status`: 执行状态(success/failed)
- `error_message`: 错误信息(如果有)
- `timestamp`: 执行时间戳

### 分析结果

1. **详细分析** (`experiment_analysis.csv`): 包含效率计算和性能指标
2. **汇总统计** (`experiment_summary.json`): 总体统计信息
3. **最优配置** (`optimal_configurations.json`): 每个任务的推荐配置

## 自定义配置

### 修改资源配置

编辑`k8s-resource-configs.yaml`文件，调整Pod的资源限制：

```yaml
resources:
  limits:
    cpu: "4"                    # CPU核心数
    memory: "2Gi"              # CPU内存
    nvidia.com/gpu: 0.5        # GPU分配比例
    nvidia.com/gpumem: 6000    # GPU内存(MB)
```

### 添加新任务

在`k8s_inference.py`中添加新的任务函数，并在`run_k8s_experiments.sh`的TASKS数组中添加任务名称。

### 修改镜像或路径

在`k8s-resource-configs.yaml`中更新：
- `image`: 容器镜像
- `hostPath`: 宿主机路径映射

## 故障排除

### 常见问题

1. **Pod创建失败**: 检查HAMI调度器是否正常运行
2. **镜像拉取失败**: 确保镜像存在且网络可达
3. **资源不足**: 检查集群是否有足够的GPU资源
4. **权限问题**: 确保kubectl有足够权限操作Pod

### 调试命令

```bash
# 查看Pod详细信息
kubectl describe pod <pod-name>

# 查看调度器日志
kubectl logs -n kube-system -l app=hami-device-plugin

# 查看GPU资源
kubectl describe node <node-name>
```

## 性能分析

分析脚本提供以下性能指标：

- **CPU效率**: 基于执行时间和CPU核心数
- **内存效率**: 基于内存使用百分比
- **GPU效率**: 基于GPU内存利用率
- **整体效率**: 加权平均效率分数

## 扩展功能

- 支持更多AI任务
- 添加网络I/O监控
- 集成Prometheus监控
- 支持多节点集群实验
- 实时结果可视化

---

**注意**: 请确保在运行实验前备份重要数据，并在非生产环境中进行测试。