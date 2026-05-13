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
    flake8 \
    mlflow

# Cria o diretório de trabalho
WORKDIR /app

# Adiciona o diretório atual ao PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV HYDRA_FULL_ERROR=1

# Cria symlink para python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set environment variables to avoid getpwuid error
ENV USER=dockeruser
ENV HOME=/tmp
ENV TORCH_HOME=/tmp/torch

# Create necessary directories and set permissions
RUN mkdir -p /tmp/.cache/torch_extensions /tmp/torch \
    && chmod -R 777 /tmp

# Copia os arquivos do projeto
COPY . /app

# Change ownership of the app directory
RUN chown -R 1000:1000 /app

# Instala o pacote em modo editável
RUN pip3 install -e .

# Expondo a porta para o Jupyter Lab (opcional)
EXPOSE 8888
EXPOSE 5000

# Comando padrão
CMD ["bash"]
