FROM python:3.8-slim

ENV CUDA_VERSION=10.2.89 \
    NVIDIA_VISIBLE_DEVICES=all \
    PATH=/root/.poetry/bin:$PATH
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg2 libgl1-mesa-dev \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \ 
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list  \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-cudart-10-2=$CUDA_VERSION-1 cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python \
    && poetry config virtualenvs.create false \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY . .
RUN poetry install
