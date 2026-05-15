# Base com CUDA 12 e Python 3
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Evita interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 to match host user and properly add to passwd
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set environment variables to avoid getpwuid error
ENV USER=appuser
ENV HOME=/home/appuser
ENV TORCH_HOME=/home/appuser/.cache/torch

# Create necessary directories and set permissions
RUN mkdir -p /home/appuser/.cache/torch_extensions /home/appuser/.cache /home/appuser/.config \
    && chown -R 1000:1000 /home/appuser

# Set MPLCONFIGDIR to avoid matplotlib permission issues
ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib

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

# Adiciona o diretório atual ao PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV HYDRA_FULL_ERROR=1

# Cria symlink para python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Change ownership of the app directory
RUN chown -R 1000:1000 /app

# Switch to appuser
USER appuser

# Create matplotlib config directory
RUN mkdir -p /home/appuser/.config/matplotlib

# Instala o pacote em modo editável
RUN pip3 install -e .

# Expondo a porta para o Jupyter Lab (opcional)
EXPOSE 8888
EXPOSE 5000

# Comando padrão
CMD ["bash"]
