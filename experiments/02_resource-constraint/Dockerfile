# 使用 NVIDIA CUDA 运行时基础镜像
FROM docker.cnb.cool/lcjd1024/os/python

# 设置工作目录
WORKDIR /app

# 避免在安装过程中的交互式提示
ENV DEBIAN_FRONTEND=noninteractive


# 配置 pip 使用清华镜像源
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf

COPY requirements.txt* ./

RUN if [ -f requirements.txt ]; then pip install -q --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; fi

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    Pillow \
    numpy \
    click \
    psutil \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./models /models

# COPY inference.py ./
COPY lena.png* ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser


ENTRYPOINT ["python", "inference.py"]

CMD ["--gpu_memory", "3GB", "--task", "SuperResolution", "--device", "cuda"]

