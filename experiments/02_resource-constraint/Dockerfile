FROM docker.cnb.cool/lcjd1024/os/python

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt* ./

RUN pip install -q --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY inference.py  inference.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "inference.py"]