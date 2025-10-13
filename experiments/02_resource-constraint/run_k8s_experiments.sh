#!/bin/bash

# K8s Resource Constraint Experiment Runner
# This script runs experiments with different resource constraints using Kubernetes Pods

set -e

# Configuration
NAMESPACE="default"
EXPERIMENT_DIR="/Users/lcjd/code-workspace/project/Project-TOMAS/experiments/02_resource-constraint"
RESULTS_DIR="${EXPERIMENT_DIR}/k8s_results"
CSV_FILE="${RESULTS_DIR}/experiment_results.csv"
YAML_FILE="${EXPERIMENT_DIR}/k8s-resource-configs.yaml"
INFERENCE_SCRIPT="${EXPERIMENT_DIR}/k8s_inference.py"

# Tasks to test
TASKS=("SuperResolution" "ImageCaptioning" "ObjectDetection")

# Create results directory
mkdir -p "$RESULTS_DIR"

# Initialize CSV file with headers
echo "pod_name,task,cpu_cores,memory_gb,gpu_ratio,gpu_memory_mb,total_time,task_execution_time,gpu_memory_used,cpu_memory_percent,status,error_message,timestamp" > "$CSV_FILE"

# Function to extract pod configurations from YAML
get_pod_configs() {
    grep "name: gpu-experiment-" "$YAML_FILE" | sed 's/.*name: //' || true
}

# Function to wait for pod completion
wait_for_pod_completion() {
    local pod_name=$1
    local timeout=300  # 5 minutes timeout
    local elapsed=0
    
    echo "等待Pod $pod_name 完成..."
    
    while [ $elapsed -lt $timeout ]; do
        local phase=$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
        
        case $phase in
            "Succeeded")
                echo "Pod $pod_name 成功完成"
                return 0
                ;;
            "Failed")
                echo "Pod $pod_name 执行失败"
                kubectl logs "$pod_name" -n "$NAMESPACE" || true
                return 1
                ;;
            "Running")
                echo -n "."
                ;;
            "NotFound")
                echo "Pod $pod_name 未找到"
                return 1
                ;;
        esac
        
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    echo "Pod $pod_name 超时"
    return 1
}

# Function to extract results from pod logs
extract_results_from_logs() {
    local pod_name=$1
    local task=$2
    local result_file="${RESULTS_DIR}/${pod_name}_${task}.json"
    
    # Get logs and try to extract JSON result
    kubectl logs "$pod_name" -n "$NAMESPACE" > "${RESULTS_DIR}/${pod_name}_${task}.log" 2>&1
    
    # Try to get the result file from the pod if possible (this might not work with hostPath)
    # For now, we'll parse the logs to extract structured information
    
    # Extract timing information from logs
    local total_time=$(grep "Total Time:" "${RESULTS_DIR}/${pod_name}_${task}.log" | sed 's/.*Total Time: \([0-9.]*\)s/\1/' || echo "0")
    local task_time=$(grep "Task Execution Time:" "${RESULTS_DIR}/${pod_name}_${task}.log" | sed 's/.*Task Execution Time: \([0-9.]*\)s/\1/' || echo "0")
    local gpu_memory=$(grep "GPU Memory Used:" "${RESULTS_DIR}/${pod_name}_${task}.log" | sed 's/.*GPU Memory Used: \([0-9.]*\)GB/\1/' || echo "0")
    local cpu_memory=$(grep "CPU Memory Used:" "${RESULTS_DIR}/${pod_name}_${task}.log" | sed 's/.*CPU Memory Used: \([0-9.]*\)%/\1/' || echo "0")
    
    # Parse pod name to extract resource configuration
    # Format: gpu-experiment-{cpu}core-{memory}-{gpu}gpu-{gpu_memory}g
    local cpu_cores=$(echo "$pod_name" | sed 's/gpu-experiment-\([0-9]*\)core-.*/\1/')
    local memory_str=$(echo "$pod_name" | sed 's/gpu-experiment-[0-9]*core-\([^-]*\)-.*/\1/')
    local gpu_ratio=$(echo "$pod_name" | sed 's/.*-\([0-9.]*\)gpu-.*/\1/')
    local gpu_memory_str=$(echo "$pod_name" | sed 's/.*gpu-\([0-9]*\)g/\1/')
    
    # Convert memory string to GB
    local memory_gb
    case $memory_str in
        *m) memory_gb=$(echo "$memory_str" | sed 's/m//' | awk '{print $1/1024}') ;;
        *g) memory_gb=$(echo "$memory_str" | sed 's/g//') ;;
        *) memory_gb="$memory_str" ;;
    esac
    
    # Convert GPU memory to MB
    local gpu_memory_mb=$((gpu_memory_str * 1000))
    
    # Check if execution was successful
    local status="success"
    local error_message=""
    if grep -q "error\|Error\|ERROR\|Exception" "${RESULTS_DIR}/${pod_name}_${task}.log"; then
        status="failed"
        error_message=$(grep -i "error\|exception" "${RESULTS_DIR}/${pod_name}_${task}.log" | head -1 | tr ',' ' ')
    fi
    
    # Get current timestamp
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Append to CSV
    echo "$pod_name,$task,$cpu_cores,$memory_gb,$gpu_ratio,$gpu_memory_mb,$total_time,$task_time,$gpu_memory,$cpu_memory,$status,\"$error_message\",$timestamp" >> "$CSV_FILE"
    
    echo "结果已保存到CSV: $pod_name, $task, ${total_time}s"
}

# Function to run experiment for a specific pod configuration and task
run_experiment() {
    local pod_name=$1
    local task=$2
    
    echo "=== 运行实验: $pod_name with task $task ==="
    
    # Create a temporary YAML file for this specific experiment
    local temp_yaml="${RESULTS_DIR}/${pod_name}_${task}.yaml"
    
    # Extract the specific pod configuration and modify for this task
    awk -v pod_name="$pod_name" -v task="$task" '
    BEGIN { in_pod = 0; new_name = pod_name "-" tolower(task) }
    /^---$/ { 
        if (in_pod) exit
        print
        next
    }
    /^apiVersion: v1$/ && !in_pod { in_pod = 1 }
    in_pod && /name: '"$pod_name"'/ { 
        sub(/name: .*/, "name: " new_name)
        actual_pod_name = new_name
    }
    in_pod && /--task.*SuperResolution/ { 
        sub(/--task.*SuperResolution/, "--task " task)
    }
    in_pod && /--task.*ImageCaptioning/ { 
        sub(/--task.*ImageCaptioning/, "--task " task) 
    }
    in_pod && /--task.*ObjectDetection/ { 
        sub(/--task.*ObjectDetection/, "--task " task)
    }
    in_pod && /command:.*python.*inference.py/ {
        sub(/\/app\/inference.py/, "/app/k8s_inference.py")
    }
    in_pod { print }
    ' "$YAML_FILE" > "$temp_yaml"
    
    local actual_pod_name="${pod_name}-$(echo $task | tr '[:upper:]' '[:lower:]')"
    
    # Delete pod if it already exists
    kubectl delete pod "$actual_pod_name" -n "$NAMESPACE" --ignore-not-found=true
    
    # Wait a moment for cleanup
    sleep 2
    
    # Apply the pod configuration
    if kubectl apply -f "$temp_yaml"; then
        echo "Pod $actual_pod_name 已创建"
        
        # Wait for pod completion
        if wait_for_pod_completion "$actual_pod_name"; then
            # Extract results
            extract_results_from_logs "$actual_pod_name" "$task"
        else
            echo "Pod $actual_pod_name 执行失败或超时"
            # Still try to extract whatever results we can
            extract_results_from_logs "$actual_pod_name" "$task"
        fi
        
        # Cleanup pod
        kubectl delete pod "$actual_pod_name" -n "$NAMESPACE" --ignore-not-found=true
    else
        echo "创建Pod $actual_pod_name 失败"
        # Log failure to CSV
        local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "$actual_pod_name,$task,,,,,0,0,0,0,failed,\"Pod creation failed\",$timestamp" >> "$CSV_FILE"
    fi
    
    # Cleanup temp file
    rm -f "$temp_yaml"
    
    echo "实验完成: $actual_pod_name"
    echo "---"
}

# Main execution
main() {
    echo "开始K8s资源约束实验"
    echo "结果将保存到: $CSV_FILE"
    echo ""
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo "错误: kubectl 未找到，请确保已安装并配置Kubernetes"
        exit 1
    fi
    
    # Check if the cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        echo "错误: 无法连接到Kubernetes集群"
        exit 1
    fi
    
    # Get all pod configurations
    local pod_configs=($(get_pod_configs))
    
    if [ ${#pod_configs[@]} -eq 0 ]; then
        echo "错误: 在 $YAML_FILE 中未找到Pod配置"
        exit 1
    fi
    
    echo "找到 ${#pod_configs[@]} 个Pod配置"
    echo "将测试 ${#TASKS[@]} 个任务: ${TASKS[*]}"
    echo ""
    
    local total_experiments=$((${#pod_configs[@]} * ${#TASKS[@]}))
    local current_experiment=0
    
    # Run experiments for each pod configuration and task combination
    for pod_config in "${pod_configs[@]}"; do
        for task in "${TASKS[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "进度: $current_experiment/$total_experiments"
            run_experiment "$pod_config" "$task"
            
            # Add a brief pause between experiments
            sleep 3
        done
    done
    
    echo ""
    echo "=== 所有实验完成 ==="
    echo "结果已保存到: $CSV_FILE"
    echo "详细日志保存在: $RESULTS_DIR"
    
    # Display summary
    echo ""
    echo "=== 实验结果摘要 ==="
    if [ -f "$CSV_FILE" ]; then
        echo "成功实验数: $(grep ",success," "$CSV_FILE" | wc -l)"
        echo "失败实验数: $(grep ",failed," "$CSV_FILE" | wc -l)"
        echo ""
        echo "CSV文件前5行:"
        head -6 "$CSV_FILE"
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi