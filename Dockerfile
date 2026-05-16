FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git sudo && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /mlflow_data && chown -R 1000:1000 /mlflow_data

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    # Core Científico e PINNs
    torch numpy scipy pandas matplotlib seaborn torchdiffeq \
    # Gestão de Experimentos e Configuração
    hydra-core mlflow pydantic \
    # Desenvolvimento e Notebooks
    jupyterlab notebook ipykernel aider-chat tqdm \
    # Qualidade de Código e Testes
    pytest black isort flake8

RUN ln -s /usr/bin/python3 /usr/bin/python

# Preparar o ambiente para o Matplotlib e Torch no Home do usuário
RUN mkdir -p /home/appuser/.cache /home/appuser/.config/matplotlib && \
    chown -R 1000:1000 /home/appuser

USER appuser
ENV PYTHONPATH=/app
ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib