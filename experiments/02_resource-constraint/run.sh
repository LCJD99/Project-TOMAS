cpu_core() {
    for i in 1 2 4 8 16 32; do
        for j in SuperResolution ImageCaptioning ObjectDetection; do
            echo $j-$i
            docker run --cpus $i\
                --gpus all \
                -e HF_HOME=/huggingface_cache \
                -v ~/.cache/huggingface:/huggingface_cache \
                -v ./inference.py:/app/inference.py \
                model-runner \
                --task $j
        done
    done
}

cpu_memory() {
    for i in 512m 800m 1g 1.5g 2g; do
        docker run --cpus 8\
            --gpus all \
            -e HF_HOME=/huggingface_cache \
            -v ~/.cache/huggingface:/huggingface_cache \
            -v ./inference.py:/app/inference.py \
            -m $i --memory-swap 4g \
            model-runner \
            --task ObjectDetection
    done
}

# cpu_memory
# cpu_core

# docker run \
#     --cpus 16 \
#     --gpus all \
#     -e LD_PRELOAD=/libvgpu/build/libvgpu.so \
#     -e CUDA_DEVICE_MEMORY_LIMIT=2048m \
#     -e CUDA_DEVICE_SM_LIMIT=5 \
#     -e HF_HOME=/huggingface_cache \
#     -v ~/.cache/huggingface:/huggingface_cache  \
#     -v ./inference.py:/app/inference.py \
#     -v ./data:/app/data \
#     -v ./vgpulock:/tmp/vgpulock \
#     model-runner \
#     python inference.py --task ObjectDetection --device cuda --gpu_memory 2GB

docker run \
  --rm \
  --gpus all \
  -e LD_PRELOAD=/libvgpu/build/libvgpu.so \
  -e HF_HOME=/huggingface_cache \
  -e GPU_CORE_UTILIZATION_POLICY=force \
  -v /home/zhangjingzhou/.cache/huggingface:/huggingface_cache \
  --cpus 16 \
  -m 2g \
  --memory-swap 8g \
  -e CUDA_DEVICE_SM_LIMIT=50 \
  -e CUDA_DEVICE_MEMORY_LIMIT=24g \
  -v /home/zhangjingzhou/tool-planning/Project-TOMAS/experiments/02_resource-constraint/data:/app/data \
  -v /home/zhangjingzhou/tool-planning/Project-TOMAS/experiments/02_resource-constraint/inference.py:/app/inference.py \
  -v /home/zhangjingzhou/tool-planning/Project-TOMAS/experiments/02_resource-constraint/qwen.py:/app/qwen.py \
  -v /home/zhangjingzhou/tool-planning/Project-TOMAS/experiments/02_resource-constraint/benchmark.py:/app/benchmark.py \
  -v /home/zhangjingzhou/tool-planning/Project-TOMAS/experiments/02_resource-constraint/vgpulock:/tmp/vgpulock \
  -v /home/zhangjingzhou/tool-planning/qwen2.5:/app/qwen2.5 \
  --pid host \
  model-runner \
  python qwen.py \
#   python inference.py \
#   --task ImageCaptioning \
#   --device cuda \
#   --gpu_memory 0.8g
