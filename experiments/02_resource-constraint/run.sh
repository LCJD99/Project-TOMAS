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
cpu_core

# docker run \
#     --cpus 2 \
#     --gpus all \
#     -it \
#     --entrypoint bash \
#     -e HF_HOME=/huggingface_cache \
#     -v ~/.cache/huggingface:/huggingface_cache  \
#     -v ./inference.py:/app/inference-dev.py \
#     model-runner \
