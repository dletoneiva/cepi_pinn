# Base com CUDA 12 e Python 3
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Evita interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

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

# Cria o diretório de trabalho
WORKDIR /app

# Expondo a porta para o Jupyter Lab (opcional)
EXPOSE 8888

# Comando padrão
CMD ["bash"]