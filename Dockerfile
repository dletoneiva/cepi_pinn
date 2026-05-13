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

# Create user with UID 1000 to match host user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Cria o diretório de trabalho
WORKDIR /app

# Adiciona o diretório atual ao PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV HYDRA_FULL_ERROR=1

# Cria symlink para python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set environment variables to avoid getpwuid error
ENV USER=appuser
ENV HOME=/home/appuser
ENV TORCH_HOME=/home/appuser/.cache/torch

# Create necessary directories and set permissions
RUN mkdir -p /home/appuser/.cache/torch_extensions /home/appuser/.cache \
    && chown -R 1000:1000 /home/appuser

# Copia os arquivos do projeto
COPY . /app

# Change ownership of the app directory
RUN chown -R 1000:1000 /app

# Switch to appuser
USER 1000

# Set up environment to avoid getpwuid error
ENV USER=appuser

# Instala o pacote em modo editável
RUN pip3 install -e .

# Expondo a porta para o Jupyter Lab (opcional)
EXPOSE 8888
EXPOSE 5000

# Comando padrão
CMD ["bash"]
