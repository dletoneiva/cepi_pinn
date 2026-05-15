# Base com CUDA 12 e Python 3
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Evita interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalação de dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git sudo && \
    rm -rf /var/lib/apt/lists/*

# Garante a criação do grupo e usuário com IDs explícitos
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Preparar o diretório de trabalho
WORKDIR /app

# Atualiza o pip e instala as libs base de Machine Learning
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    pydantic \
    jupyterlab \
    notebook \
    ipykernel \
    aider-chat \
    hydra-core \
    mlflow \
    torchdiffeq \
    pytest \
    black \
    isort \
    flake8


# Cria symlink para python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Switch to appuser
USER appuser

# Create matplotlib config directory
RUN mkdir -p /home/appuser/.config/matplotlib

ENV PYTHONPATH=/app

