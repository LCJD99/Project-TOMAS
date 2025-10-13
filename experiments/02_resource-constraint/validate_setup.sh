#!/bin/bash

# Quick test script to validate K8s experiment setup
# This script performs basic validation before running the full experiment

set -e

EXPERIMENT_DIR="/Users/lcjd/code-workspace/project/Project-TOMAS/experiments/02_resource-constraint"
YAML_FILE="${EXPERIMENT_DIR}/k8s-resource-configs.yaml"

echo "=== K8s资源约束实验设置验证 ==="
echo ""

# Check if kubectl is available
echo "1. 检查kubectl可用性..."
if command -v kubectl &> /dev/null; then
    echo "✓ kubectl已安装"
    kubectl version --client --short
else
    echo "✗ kubectl未找到，请安装并配置Kubernetes"
    exit 1
fi

# Check cluster connectivity
echo ""
echo "2. 检查集群连接..."
if kubectl cluster-info &> /dev/null; then
    echo "✓ 集群连接正常"
    kubectl cluster-info | head -2
else
    echo "✗ 无法连接到Kubernetes集群"
    exit 1
fi

# Check HAMI scheduler
echo ""
echo "3. 检查HAMI调度器..."
if kubectl get pods -n kube-system | grep -q hami; then
    echo "✓ HAMI调度器运行中"
    kubectl get pods -n kube-system | grep hami
else
    echo "⚠ 未找到HAMI调度器，vGPU功能可能不可用"
fi

# Check GPU resources
echo ""
echo "4. 检查GPU资源..."
gpu_nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}' | xargs -n1 kubectl describe node | grep -A 1 "nvidia.com/gpu" | grep -v "Allocated" | wc -l)
if [ "$gpu_nodes" -gt 0 ]; then
    echo "✓ 发现GPU资源"
    kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
else
    echo "⚠ 未发现GPU资源，请确保节点已安装GPU插件"
fi

# Check YAML configuration
echo ""
echo "5. 检查YAML配置文件..."
if [ -f "$YAML_FILE" ]; then
    echo "✓ 配置文件存在: $YAML_FILE"
    pod_count=$(grep -c "name: gpu-experiment-" "$YAML_FILE")
    echo "✓ 发现 $pod_count 个Pod配置"
else
    echo "✗ 配置文件不存在: $YAML_FILE"
    exit 1
fi

# Validate YAML syntax
echo ""
echo "6. 验证YAML语法..."
if kubectl apply --dry-run=client -f "$YAML_FILE" &> /dev/null; then
    echo "✓ YAML语法正确"
else
    echo "✗ YAML语法错误"
    kubectl apply --dry-run=client -f "$YAML_FILE"
    exit 1
fi

# Check required files
echo ""
echo "7. 检查必需文件..."
required_files=(
    "${EXPERIMENT_DIR}/k8s_inference.py"
    "${EXPERIMENT_DIR}/run_k8s_experiments.sh" 
    "${EXPERIMENT_DIR}/analyze_results.py"
    "${EXPERIMENT_DIR}/lena.png"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $(basename "$file")"
    else
        echo "✗ 缺少文件: $(basename "$file")"
        exit 1
    fi
done

# Check HuggingFace cache (optional)
echo ""
echo "8. 检查HuggingFace缓存..."
hf_cache_dir="/home/user/.cache/huggingface"
if [ -d "$hf_cache_dir" ]; then
    echo "✓ HuggingFace缓存目录存在: $hf_cache_dir"
    model_count=$(find "$hf_cache_dir" -name "*.bin" 2>/dev/null | wc -l)
    echo "  发现 $model_count 个模型文件"
else
    echo "⚠ HuggingFace缓存目录不存在，首次运行可能需要下载模型"
fi

# Test a single pod creation (dry run)
echo ""
echo "9. 测试Pod创建 (dry-run)..."
temp_yaml="/tmp/test-pod.yaml"
head -n 50 "$YAML_FILE" > "$temp_yaml"
echo "---" >> "$temp_yaml"

if kubectl apply --dry-run=server -f "$temp_yaml" &> /dev/null; then
    echo "✓ Pod创建测试通过"
else
    echo "⚠ Pod创建测试失败，可能的问题:"
    kubectl apply --dry-run=server -f "$temp_yaml" 2>&1 | head -5
fi

rm -f "$temp_yaml"

# Summary
echo ""
echo "=== 验证摘要 ==="
echo "✓ kubectl配置正确"
echo "✓ 集群连接正常"
echo "✓ 配置文件有效"
echo "✓ 必需文件齐全"
echo ""
echo "🚀 系统已准备就绪，可以运行实验!"
echo ""
echo "运行实验命令:"
echo "  cd $EXPERIMENT_DIR"
echo "  chmod +x run_k8s_experiments.sh"
echo "  ./run_k8s_experiments.sh"
echo ""
echo "分析结果命令:"
echo "  pip install pandas numpy  # 如果尚未安装"
echo "  python analyze_results.py"